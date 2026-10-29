#!/usr/bin/env python3
"""
Ball Balancing Robot - single-file controller + live-tuning web UI
===================================================================
3-RRS platform | Raspberry Pi 4 | Pi Camera v1.3 | PCA9685 + 3 servos

Install on the Pi:
    sudo apt install -y python3-picamera2 python3-opencv python3-numpy python3-flask
    pip3 install --break-system-packages adafruit-circuitpython-servokit
"""
import argparse
import copy
import json
import logging
import math
import os
import signal
import socket
import threading
import time
from collections import deque

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

log = logging.getLogger("balancer")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(APP_DIR, "balancer_config.json")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Servo "limit_down" / "limit_up": the two hard end-stops (degrees) of each servo.
#   limit_down = the angle that puts the platform LOWEST   (your "min")
#   limit_up   = the angle that puts the platform HIGHEST  (your "max")
# The software never commands a servo outside [min(down, up), max(down, up)].
#
# All camera pixel values (center, target, detection_radius, tracking radii) are in
# "work" resolution pixels (320x240 by default), not capture resolution.
DEFAULT_CONFIG = {
    "geometry": {"LP": 7.62, "L1": 8.5, "L2": 4.5, "LB": 4.0, "invert": False},
    "servo": {
        "channels": [14, 15, 13],      # PCA9685 channels for servo 1, 2, 3
        "actuation_range": 180,
        "pulse_min": 500,
        "pulse_max": 2500,
        "offsets": [0.0, 0.0, 0.0],
        "limit_down": [90.0, 110.0, 105.0],
        "limit_up": [25.0, 40.0, 30.0],
    },
    "platform": {"h_work": 10.6},
    "pid": {
        "kp": 0.0063, "ki": 0.0006, "kd": 0.0060,
        "alpha": 0.65,          # output filter  alpha*new + (1-alpha)*old
        "beta": 0.3,            # tilt = tilt_limit * tanh(beta * |pid|)
        "tilt_limit": 10.0,     # max tilt in degrees the PID may ask for
        "i_limit": 0.15,        # max contribution of the integral term (anti-windup)
        "rot_deg": 90.0,        # rotation from image axes to platform axes
        "mirror": False,        # flip image Y before rotating
        "lost_timeout": 0.3,    # seconds without ball before the platform levels itself
    },
    "camera": {
        "capture": [640, 480],
        "work": [320, 240],
        "center": [160.0, 120.0],
        "target": [160.0, 120.0],
        "detection_radius": 105.0,
        "px_per_cm": 14.0,      # work pixels per cm on the platform surface
        "ball_diameter_cm": 2.8,
        "exposure_auto": True,
        "exposure_us": 6000,
        "gain": 4.0,
        "awb_auto": True,       # turn off to lock white balance so colours stay stable
    },
    "tracking": {   # dark green ball, OpenCV HSV (H 0-179)
        "h_min": 35, "h_max": 90,
        "s_min": 60, "s_max": 255,
        "v_min": 20, "v_max": 200,
        "blur": 3, "erode": 1, "dilate": 1,
        "min_radius": 6.0, "max_radius": 45.0,
        "min_circularity": 0.6,
    },
}

RANGES = {
    "pid.kp": (0, 1), "pid.ki": (0, 1), "pid.kd": (0, 1),
    "pid.alpha": (0.01, 1.0), "pid.beta": (0.0, 20.0),
    "pid.tilt_limit": (0.5, 30.0), "pid.i_limit": (0.0, 5.0),
    "pid.rot_deg": (-360, 360), "pid.lost_timeout": (0.05, 3.0),
    "platform.h_work": (3.0, 30.0),
    "servo.offsets": (-45.0, 45.0),
    "servo.limit_down": (0.0, 180.0), "servo.limit_up": (0.0, 180.0),
    "camera.detection_radius": (5.0, 400.0),
    "camera.exposure_us": (100, 30000), "camera.gain": (1.0, 16.0),
    "camera.px_per_cm": (1.0, 200.0), "camera.ball_diameter_cm": (0.5, 10.0),
    "tracking.h_min": (0, 179), "tracking.h_max": (0, 179),
    "tracking.s_min": (0, 255), "tracking.s_max": (0, 255),
    "tracking.v_min": (0, 255), "tracking.v_max": (0, 255),
    "tracking.blur": (1, 15), "tracking.erode": (0, 6), "tracking.dilate": (0, 6),
    "tracking.min_radius": (1.0, 300.0), "tracking.max_radius": (2.0, 300.0),
    "tracking.min_circularity": (0.0, 1.0),
}


def _coerce(current, v):
    if isinstance(current, bool):
        return bool(v)
    if isinstance(current, int):
        return int(round(float(v)))
    if isinstance(current, float):
        return float(v)
    if isinstance(current, list):
        if not isinstance(v, (list, tuple)) or len(v) != len(current):
            raise ValueError("bad list")
        return [_coerce(c, x) for c, x in zip(current, v)]
    return v


def _clamp_range(path, v):
    r = RANGES.get(path)
    if r is None:
        return v
    if isinstance(v, list):
        return [min(max(x, r[0]), r[1]) for x in v]
    if isinstance(v, bool):
        return v
    out = min(max(v, r[0]), r[1])
    return int(out) if isinstance(v, int) else out


class Config:
    def __init__(self, path):
        self.path = path
        self.lock = threading.RLock()
        self.data = copy.deepcopy(DEFAULT_CONFIG)
        if os.path.isfile(path):
            try:
                with open(path) as f:
                    self._apply(self.data, json.load(f), "", set())
                log.info("Loaded settings from %s", path)
            except Exception as e:  # noqa
                log.error("Could not read %s (%s) - using defaults", path, e)

    def _apply(self, dst, src, prefix, changed):
        for k, v in src.items():
            if k not in dst:
                continue
            p = f"{prefix}.{k}" if prefix else k
            if isinstance(dst[k], dict):
                if isinstance(v, dict):
                    self._apply(dst[k], v, p, changed)
                continue
            try:
                nv = _clamp_range(p, _coerce(dst[k], v))
            except (ValueError, TypeError):
                continue
            if nv != dst[k]:
                dst[k] = nv
                changed.add(p)

    def patch(self, patch):
        changed = set()
        with self.lock:
            self._apply(self.data, patch, "", changed)
            cam = self.data["camera"]
            w, h = cam["work"]
            for key in ("center", "target"):
                cam[key] = [min(max(cam[key][0], 0.0), float(w)), min(max(cam[key][1], 0.0), float(h))]
        return changed

    def save(self):
        with self.lock:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.data, f, indent=2)
            os.replace(tmp, self.path)


# ---------------------------------------------------------------------------
# Named tuning presets (PID + ball colour/size), stored in balancer_presets.json
# ---------------------------------------------------------------------------
# Camera rotation / mirror are NOT part of a preset: they belong to the hardware, not to a ball.
PRESET_KEYS = {
    "pid": ["kp", "ki", "kd", "alpha", "beta", "tilt_limit", "i_limit", "lost_timeout"],
    "tracking": ["h_min", "h_max", "s_min", "s_max", "v_min", "v_max", "blur", "erode", "dilate",
                 "min_radius", "max_radius", "min_circularity"],
    "camera": ["ball_diameter_cm"],
}


def preset_values(config):
    out = {}
    with config.lock:
        for sec, keys in PRESET_KEYS.items():
            out[sec] = {k: copy.deepcopy(config.data[sec][k]) for k in keys}
    return out


class PresetStore:
    def __init__(self, path):
        self.path = path
        self.lock = threading.RLock()
        self.items = {}
        if os.path.isfile(path):
            try:
                with open(path) as f:
                    raw = json.load(f).get("presets", {})
                for name, item in raw.items():
                    if isinstance(item, dict) and isinstance(item.get("values"), dict):
                        self.items[str(name)] = {"saved": str(item.get("saved", "")), "values": item["values"]}
                log.info("Loaded %d tuning presets from %s", len(self.items), path)
            except Exception as e:  # noqa
                log.error("Could not read %s (%s) - starting with no presets", path, e)

    @staticmethod
    def clean(name):
        return " ".join(str(name or "").split())[:40]

    def _flush(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"version": 1, "presets": self.items}, f, indent=2)
        os.replace(tmp, self.path)

    def listing(self):
        with self.lock:
            return [{"name": n, "saved": it["saved"], "values": it["values"]}
                    for n, it in sorted(self.items.items(), key=lambda kv: kv[0].lower())]

    def get(self, name):
        with self.lock:
            it = self.items.get(self.clean(name))
            return copy.deepcopy(it["values"]) if it else None

    def save(self, name, values, overwrite):
        n = self.clean(name)
        if not n:
            return False, "Give the preset a name"
        with self.lock:
            if n in self.items and not overwrite:
                return False, "exists"
            self.items[n] = {"saved": time.strftime("%Y-%m-%d %H:%M:%S"), "values": values}
            try:
                self._flush()
            except OSError as e:
                return False, f"Could not write presets file: {e}"
        return True, None

    def delete(self, name):
        n = self.clean(name)
        with self.lock:
            if n not in self.items:
                return False, "No such preset"
            del self.items[n]
            try:
                self._flush()
            except OSError as e:
                return False, f"Could not write presets file: {e}"
        return True, None

    def rename(self, old, new):
        o, n = self.clean(old), self.clean(new)
        if not n:
            return False, "Give the preset a name"
        with self.lock:
            if o not in self.items:
                return False, "No such preset"
            if n != o and n in self.items:
                return False, "A preset with that name already exists"
            self.items[n] = self.items.pop(o)
            try:
                self._flush()
            except OSError as e:
                return False, f"Could not write presets file: {e}"
        return True, None


# ---------------------------------------------------------------------------
# Kinematics (3-RRS platform) - ported from signag/Ball-Balancing-Robot_CS
# ---------------------------------------------------------------------------
class Kinematics:
    def __init__(self, lp, l1, l2, lb, invert=False):
        self.lp, self.l1, self.l2, self.lb, self.invert = lp, l1, l2, lb, invert
        self.max_h = math.sqrt((l1 + l2) ** 2 - (lp - lb) ** 2) - 0.2
        if l1 > l2:
            min_h = math.sqrt(l1 ** 2 - (lb + l2 - lp) ** 2)
        elif l2 > l1:
            min_h = math.sqrt((l2 - l1) ** 2 - (lp - lb) ** 2)
        else:
            min_h = 0.0
        self.min_h = min_h + 0.45
        if self.min_h >= self.max_h:
            raise ValueError("geometry leaves no usable height range")
        self._tm = {}

    def solve(self, a, b, c, h):
        """Platform normal (a,b,c) and centre height h -> arm angles theta1..3 (radians)."""
        lp, lb, l1, l2 = self.lp, self.lb, self.l1, self.l2
        s3 = math.sqrt(3)
        d1 = math.sqrt(4 * c ** 2 + (a - s3 * b) ** 2)
        d2 = math.sqrt(c ** 2 + a ** 2)
        d3 = math.sqrt(4 * c ** 2 + (a + s3 * b) ** 2)
        A1 = [-lp * c / d1, s3 * lp * c / d1, h + (a - s3 * b) * lp / d1]
        A2 = [lp * c / d2, 0.0, h - lp * a / d2]
        A3 = [-lp * c / d3, -s3 * lp * c / d3, h + (a + s3 * b) * lp / d3]

        a11, a12, a13 = A1
        a21, _, a23 = A2
        a31, a32, a33 = A3

        p1 = (-a11 + s3 * a12 - 2 * lb) / a13
        q1 = (a11 ** 2 + a12 ** 2 + a13 ** 2 + l2 ** 2 - l1 ** 2 - lb ** 2) / (2 * a13)
        r1, s1, t1 = p1 ** 2 + 4, 2 * p1 * q1 + 4 * lb, q1 ** 2 + lb ** 2 - l2 ** 2

        p2 = (lb - a21) / a23
        q2 = (a21 ** 2 + a23 ** 2 - lb ** 2 + l2 ** 2 - l1 ** 2) / (2 * a23)
        r2, s2, t2 = p2 ** 2 + 1, 2 * (p2 * q2 - lb), q2 ** 2 - l2 ** 2 + lb ** 2

        p3 = (-a31 - s3 * a32 - 2 * lb) / a33
        q3 = (a31 ** 2 + a32 ** 2 + a33 ** 2 + l2 ** 2 - l1 ** 2 - lb ** 2) / (2 * a33)
        r3, s3_, t3 = p3 ** 2 + 4, 2 * p3 * q3 + 4 * lb, q3 ** 2 + lb ** 2 - l2 ** 2

        sign = -1.0 if self.invert else 1.0

        c11 = (-s1 - math.sqrt(s1 ** 2 - 4 * r1 * t1)) / (2 * r1)
        c12 = -s3 * c11
        c13 = sign * math.sqrt(l2 ** 2 - 4 * c11 ** 2 - 4 * lb * c11 - lb ** 2)

        c21 = (-s2 + math.sqrt(s2 ** 2 - 4 * r2 * t2)) / (2 * r2)
        c23 = sign * math.sqrt(l2 ** 2 - (c21 - lb) ** 2)

        c31 = (-s3_ - math.sqrt(s3_ ** 2 - 4 * r3 * t3)) / (2 * r3)
        c32 = s3 * c31
        c33 = sign * math.sqrt(l2 ** 2 - 4 * c31 ** 2 - 4 * lb * c31 - lb ** 2)

        th1 = math.pi / 2 - math.atan2(math.sqrt(c11 ** 2 + c12 ** 2) - lb, c13)
        th2 = math.atan2(c23, c21 - lb)
        th3 = math.pi / 2 - math.atan2(math.sqrt(c31 ** 2 + c32 ** 2) - lb, c33)
        return th1, th2, th3

    def _valid(self, theta, h):
        c = math.cos(theta)
        for s in (1, -1):
            a21 = self.lp * c
            a23 = h - self.lp * (s * math.sin(theta))
            try:
                p2 = (self.lb - a21) / a23
                q2 = (a21 ** 2 + a23 ** 2 - self.lb ** 2 + self.l2 ** 2 - self.l1 ** 2) / (2 * a23)
                r2 = p2 ** 2 + 1
                s2 = 2 * (p2 * q2 - self.lb)
                t2 = q2 ** 2 - self.l2 ** 2 + self.lb ** 2
                disc = s2 ** 2 - 4 * r2 * t2
                if disc < 0:
                    return False
                c21 = (-s2 + math.sqrt(disc)) / (2 * r2)
                delta = self.l2 ** 2 - (c21 - self.lb) ** 2
                if delta < 0:
                    return False
                c23 = math.sqrt(delta)
                if abs(math.hypot(a21 - c21, a23 - c23) - self.l1) > 1e-3:
                    return False
                if abs(math.hypot(self.lb - c21, c23) - self.l2) > 1e-3:
                    return False
            except (ValueError, ZeroDivisionError):
                return False
        return True

    def max_theta(self, h):
        """Largest platform tilt (degrees) that is reachable at height h."""
        key = round(h, 2)
        if key in self._tm:
            return self._tm[key]
        lo, hi = 0.0, math.radians(20)
        while hi - lo > 1e-3:
            mid = (lo + hi) / 2
            if self._valid(mid, h):
                lo = mid
            else:
                hi = mid
        val = max(0.0, math.degrees(round(lo, 4)) - 0.5)
        self._tm[key] = val
        return val


# ---------------------------------------------------------------------------
# Servo drivers
# ---------------------------------------------------------------------------
class ServoDriver:
    """Real PCA9685 servos through adafruit_servokit."""

    def __init__(self, scfg):
        from adafruit_servokit import ServoKit
        self.kit = ServoKit(channels=16)
        self.servos = [self.kit.servo[ch] for ch in scfg["channels"]]
        for s in self.servos:
            s.actuation_range = scfg["actuation_range"]
            s.set_pulse_width_range(scfg["pulse_min"], scfg["pulse_max"])
        self.last = [None] * 3

    def set_angle(self, i, angle):
        if self.last[i] is not None and abs(self.last[i] - angle) < 0.05:
            return
        self.servos[i].angle = float(angle)
        self.last[i] = angle


class MockDriver:
    def __init__(self):
        self.last = [None] * 3

    def set_angle(self, i, angle):
        self.last[i] = angle


# ---------------------------------------------------------------------------
# Platform: pose -> servo angles, offsets, hard limits
# ---------------------------------------------------------------------------
class Platform:
    def __init__(self, cfg, driver):
        self.cfg = cfg
        self.driver = driver
        self.lock = threading.RLock()
        self.theta = 0.0
        self.phi = 0.0
        self.h = cfg.data["platform"]["h_work"]
        self.angles = [None, None, None]
        self.clamped = [False, False, False]
        self.kin = None
        self.h_range = (0.0, 0.0)
        err = self.rebuild()
        if err:
            raise RuntimeError(err)
        self.set_pose(0.0, 0.0, self.h_work())

    # -- setup -----------------------------------------------------------
    def rebuild(self):
        g = self.cfg.data["geometry"]
        try:
            kin = Kinematics(g["LP"], g["L1"], g["L2"], g["LB"], g["invert"])
        except (ValueError, ZeroDivisionError) as e:
            return f"Invalid geometry: {e}"
        with self.lock:
            self.kin = kin
            self.h_range = self._compute_h_range()
        return None

    def limits(self):
        s = self.cfg.data["servo"]
        return [(min(d, u), max(d, u)) for d, u in zip(s["limit_down"], s["limit_up"])]

    def _level_angles(self, h):
        th = self.kin.solve(0.0, 0.0, 1.0, h)
        offs = self.cfg.data["servo"]["offsets"]
        return [90.0 - math.degrees(t) + o for t, o in zip(th, offs)]

    def _compute_h_range(self):
        """Heights at which the level platform is reachable inside the servo limits."""
        lo_k, hi_k = self.kin.min_h, self.kin.max_h
        lims = self.limits()
        ok = []
        n = 60
        for i in range(n + 1):
            h = lo_k + (hi_k - lo_k) * i / n
            try:
                ang = self._level_angles(h)
            except (ValueError, ZeroDivisionError):
                continue
            if all(l <= a <= u for a, (l, u) in zip(ang, lims)):
                ok.append(h)
        if ok:
            return (min(ok), max(ok))
        return (lo_k, hi_k)

    def h_work(self):
        lo, hi = self.h_range
        return min(max(self.cfg.data["platform"]["h_work"], lo), hi)

    # -- commands --------------------------------------------------------
    def pose_angles(self, theta, phi, h):
        lo, hi = self.h_range
        h = min(max(h, lo), hi)
        theta = min(max(theta, 0.0), self.kin.max_theta(h))
        phi = phi % 360.0
        tr, pr = math.radians(theta), math.radians(phi)
        a, b, c = math.sin(tr) * math.cos(pr), math.sin(tr) * math.sin(pr), math.cos(tr)
        th = self.kin.solve(a, b, c, h)
        offs = self.cfg.data["servo"]["offsets"]
        raw = [90.0 - math.degrees(t) + o for t, o in zip(th, offs)]
        return raw, theta, phi, h

    def _write(self, raw):
        out, flags = [], []
        for r, (lo, hi) in zip(raw, self.limits()):
            v = min(max(r, lo), hi)
            out.append(v)
            flags.append(abs(v - r) > 0.01)
        for i, v in enumerate(out):
            self.driver.set_angle(i, v)
        self.angles, self.clamped = out, flags

    def set_pose(self, theta, phi, h):
        with self.lock:
            try:
                raw, theta, phi, h = self.pose_angles(theta, phi, h)
            except (ValueError, ZeroDivisionError):
                return False
            self._write(raw)
            self.theta, self.phi, self.h = theta, phi, h
            return True

    def ramp_to_pose(self, theta, phi, h, dur=0.6):
        """Smoothly move (in servo-angle space) from wherever the servos are to a pose."""
        with self.lock:
            raw, theta, phi, h = self.pose_angles(theta, phi, h)
            target = [min(max(r, lo), hi) for r, (lo, hi) in zip(raw, self.limits())]
            start = [a if a is not None else t for a, t in zip(self.angles, target)]
        n = max(2, int(dur * 40))
        for i in range(1, n + 1):
            u = i / n
            u = u * u * (3 - 2 * u)
            with self.lock:
                self._write([s + (t - s) * u for s, t in zip(start, target)])
            time.sleep(dur / n)
        with self.lock:
            self.theta, self.phi, self.h = theta, phi, h

    def set_servo(self, i, angle):
        with self.lock:
            lo, hi = self.limits()[i]
            v = min(max(angle, lo), hi)
            self.driver.set_angle(i, v)
            self.angles[i] = v
            self.clamped[i] = abs(v - angle) > 0.01

    def set_servo_free(self, i, angle):
        """Used only while the user is dragging a limit slider (absolute 0-180 guard)."""
        with self.lock:
            v = min(max(angle, 0.0), 180.0)
            self.driver.set_angle(i, v)
            self.angles[i] = v
            self.clamped[i] = False


# ---------------------------------------------------------------------------
# Ball tracking
# ---------------------------------------------------------------------------
def hsv_mask(hsv, t):
    """HSV range mask. If h_min > h_max the hue range wraps around 179 -> 0 (needed for red)."""
    h0, h1 = int(t["h_min"]), int(t["h_max"])
    s0, s1, v0, v1 = int(t["s_min"]), int(t["s_max"]), int(t["v_min"]), int(t["v_max"])
    if h0 <= h1:
        return cv2.inRange(hsv, np.array([h0, s0, v0], np.uint8), np.array([h1, s1, v1], np.uint8))
    a = cv2.inRange(hsv, np.array([h0, s0, v0], np.uint8), np.array([179, s1, v1], np.uint8))
    b = cv2.inRange(hsv, np.array([0, s0, v0], np.uint8), np.array([h1, s1, v1], np.uint8))
    return cv2.bitwise_or(a, b)


class Detection:
    __slots__ = ("found", "x", "y", "r", "candidates", "mask")

    def __init__(self):
        self.found = False
        self.x = self.y = self.r = 0.0
        self.candidates = []   # [(contour, status)]  status: ok | size | shape | outside
        self.mask = None


class Tracker:
    def detect(self, small, t, center, radius):
        det = Detection()
        k = int(t["blur"]) | 1
        img = cv2.GaussianBlur(small, (k, k), 0) if k > 1 else small
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = hsv_mask(hsv, t)
        if t["erode"] > 0:
            mask = cv2.erode(mask, None, iterations=int(t["erode"]))
        if t["dilate"] > 0:
            mask = cv2.dilate(mask, None, iterations=int(t["dilate"]))
        det.mask = mask

        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        best_area = 0.0
        for c in contours:
            area = cv2.contourArea(c)
            if area < 12:
                continue
            r = math.sqrt(area / math.pi)
            status = "ok"
            if r < t["min_radius"] or r > t["max_radius"]:
                status = "size"
            else:
                per = cv2.arcLength(c, True)
                circ = 4 * math.pi * area / (per * per) if per > 0 else 0.0
                if circ < t["min_circularity"]:
                    status = "shape"
                else:
                    m = cv2.moments(c)
                    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
                    if math.hypot(cx - center[0], cy - center[1]) > radius:
                        status = "outside"
                    elif area > best_area:
                        best_area = area
                        det.found, det.x, det.y, det.r = True, cx, cy, r
            det.candidates.append((c, status))
        return det


# ---------------------------------------------------------------------------
# PID
# ---------------------------------------------------------------------------
class PID:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sx = self.sy = 0.0
        self.px = self.py = None
        self.ox = self.oy = 0.0
        self.last_t = None

    def step(self, err_img, now, p, norm, tilt_cap):
        ex, ey = err_img
        if p["mirror"]:
            ey = -ey
        a = math.radians(p["rot_deg"])
        ca, sa = math.cos(a), math.sin(a)
        x = (ex * ca - ey * sa) * norm      # error in platform axes, "200 px wide frame" units
        y = (ex * sa + ey * ca) * norm

        dt = 0.001 if self.last_t is None else min(max(now - self.last_t, 1e-3), 0.1)
        self.last_t = now

        self.sx += x * dt
        self.sy += y * dt
        if p["ki"] > 0:
            lim = p["i_limit"] / p["ki"]
            self.sx = min(max(self.sx, -lim), lim)
            self.sy = min(max(self.sy, -lim), lim)
        dx = 0.0 if self.px is None else (x - self.px) / dt
        dy = 0.0 if self.py is None else (y - self.py) / dt
        self.px, self.py = x, y

        ux = p["kp"] * x + p["ki"] * self.sx + p["kd"] * dx
        uy = p["kp"] * y + p["ki"] * self.sy + p["kd"] * dy
        al = p["alpha"]
        fx = al * ux + (1 - al) * self.ox
        fy = al * uy + (1 - al) * self.oy
        self.ox, self.oy = fx, fy

        # Ball sits at +error -> tilt so that it rolls back: azimuth points against the error
        phi = (math.degrees(math.atan2(fy, fx)) + 180.0) % 360.0
        r = math.hypot(fx, fy)
        theta = min(p["tilt_limit"] * math.tanh(p["beta"] * r), tilt_cap)
        info = {
            "err": [x, y], "p": [p["kp"] * x, p["kp"] * y],
            "i": [p["ki"] * self.sx, p["ki"] * self.sy],
            "d": [p["kd"] * dx, p["kd"] * dy],
            "out": r, "theta": theta, "phi": phi,
        }
        return theta, phi, info


# ---------------------------------------------------------------------------
# Target paths: the PID target moves along a shape around the target point
# ---------------------------------------------------------------------------
class TargetPattern:
    """Offset (cm, image axes: x right, y down) added to the target point.
    Phases: idle -> approach (glide to the start of the path) -> run -> return (glide back) -> idle."""
    SHAPES = ("circle", "square", "triangle", "star", "line", "spiral")

    def __init__(self):
        self.lock = threading.RLock()
        self.shape, self.size, self.speed, self.direction, self.rot = "circle", 2.5, 3.0, 1, 0.0
        self.phase = "idle"
        self.off = np.zeros(2)
        self.s = 0.0                 # distance travelled along the path (cm)
        self.sp_theta, self.sp_r, self.sp_dir = 0.0, 0.0, 1.0

    @property
    def active(self):
        return self.phase != "idle"

    def configure(self, d):
        with self.lock:
            if d.get("shape") in self.SHAPES:
                self.shape = d["shape"]
            if "size" in d:
                self.size = min(max(float(d["size"]), 0.3), 8.0)
            if "speed" in d:
                self.speed = min(max(float(d["speed"]), 0.3), 15.0)
            if "rot" in d:
                self.rot = float(d["rot"]) % 360.0
            if "direction" in d:
                self.direction = -1 if d["direction"] in ("ccw", -1, "-1") else 1

    def start(self):
        with self.lock:
            self.s = 0.0
            self._reset_spiral()
            self.phase = "approach"

    def stop(self):
        with self.lock:
            if self.phase != "idle":
                self.phase = "return"

    def reset(self):
        with self.lock:
            self.phase = "idle"
            self.off = np.zeros(2)
            self.s = 0.0

    def _rmin(self):
        return min(0.8, self.size * 0.4)

    def _reset_spiral(self):
        self.sp_theta = math.radians(self.rot) - math.pi / 2
        self.sp_r = self._rmin()
        self.sp_dir = 1.0

    def _rotate(self, x, y):
        a = math.radians(self.rot)
        return (x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a))

    def _vertices(self):
        sz = self.size
        if self.shape == "square":
            pts = [(sz * math.cos(math.radians(45 + 90 * k)), sz * math.sin(math.radians(45 + 90 * k))) for k in range(4)]
        elif self.shape == "triangle":
            pts = [(sz * math.cos(math.radians(-90 + 120 * k)), sz * math.sin(math.radians(-90 + 120 * k))) for k in range(3)]
        elif self.shape == "star":
            pts = []
            for k in range(5):
                a, b = math.radians(-90 + 72 * k), math.radians(-54 + 72 * k)
                pts.append((sz * math.cos(a), sz * math.sin(a)))
                pts.append((0.4 * sz * math.cos(b), 0.4 * sz * math.sin(b)))
        else:   # line: back and forth
            pts = [(-sz, 0.0), (sz, 0.0)]
        pts = [self._rotate(x, y) for x, y in pts]
        return pts[::-1] if self.direction < 0 else pts

    def _point(self):
        if self.shape == "circle":
            th = math.radians(self.rot) - math.pi / 2 + self.direction * self.s / max(self.size, 0.3)
            return np.array([self.size * math.cos(th), self.size * math.sin(th)])
        if self.shape == "spiral":
            return self.sp_r * np.array([math.cos(self.sp_theta), math.sin(self.sp_theta)])
        pts = self._vertices()
        n = len(pts)
        seg = [math.dist(pts[i], pts[(i + 1) % n]) for i in range(n)]
        total = sum(seg)
        s = self.s % total
        for i in range(n):
            if s <= seg[i] or i == n - 1:
                u = s / seg[i] if seg[i] > 0 else 0.0
                p, q = pts[i], pts[(i + 1) % n]
                return np.array([p[0] + (q[0] - p[0]) * u, p[1] + (q[1] - p[1]) * u])
            s -= seg[i]
        return np.zeros(2)

    @staticmethod
    def _slew(cur, tgt, step):
        diff = tgt - cur
        d = float(np.hypot(*diff))
        if d <= step:
            return tgt.copy(), 0.0
        return cur + diff / d * step, d - step

    def update(self, dt, paused=False):
        with self.lock:
            if self.phase == "idle":
                return self.off.copy()
            dt = min(max(dt, 0.0), 0.1)
            v = self.speed
            if self.phase == "approach":
                self.off, rem = self._slew(self.off, self._point(), 0.0 if paused else v * dt)
                if rem < 0.05 and not paused:
                    self.phase = "run"
            elif self.phase == "run":
                if not paused:
                    self.s += v * dt
                    if self.shape == "spiral":
                        self.sp_r += self.sp_dir * v * 0.06 * dt
                        if self.sp_r >= self.size:
                            self.sp_r, self.sp_dir = self.size, -1.0
                        elif self.sp_r <= self._rmin():
                            self.sp_r, self.sp_dir = self._rmin(), 1.0
                        self.sp_theta += self.direction * (v / max(self.sp_r, 0.8)) * dt
                self.off, _ = self._slew(self.off, self._point(), 3.0 * v * dt)
            else:   # return
                self.off, rem = self._slew(self.off, np.zeros(2), v * dt)
                if rem < 0.02:
                    self.phase = "idle"
                    self.off = np.zeros(2)
            return self.off.copy()

    def preview(self):
        with self.lock:
            if self.shape in ("circle", "spiral"):
                return [(self.size * math.cos(t), self.size * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 73)]
            pts = self._vertices()
            return pts + [pts[0]] if self.shape != "line" else pts

    def state(self):
        with self.lock:
            return {"active": self.phase != "idle", "phase": self.phase, "shape": self.shape, "size": self.size,
                    "speed": self.speed, "direction": "ccw" if self.direction < 0 else "cw", "rot": self.rot}


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------
class PiCamera:
    def __init__(self, cfg):
        from picamera2 import Picamera2
        self.cfg = cfg
        w, h = cfg.data["camera"]["capture"]
        self.picam2 = Picamera2()
        kw = dict(
            main={"size": (w, h), "format": "RGB888"},   # RGB888 is BGR byte order = OpenCV order
            controls={"FrameDurationLimits": (16667, 16667)},
            buffer_count=4,
        )
        try:
            conf = self.picam2.create_preview_configuration(sensor={"output_size": (w, h), "bit_depth": 10}, **kw)
        except Exception as e:  # noqa
            log.warning("Could not pin the sensor mode (%s); using default selection", e)
            conf = self.picam2.create_preview_configuration(**kw)
        self.picam2.configure(conf)
        self.picam2.start()
        self.apply_controls()

    def apply_controls(self):
        c = self.cfg.data["camera"]
        self.picam2.set_controls({"AwbEnable": bool(c["awb_auto"])})
        if c["exposure_auto"]:
            self.picam2.set_controls({"AeEnable": True})
        else:
            self.picam2.set_controls({"AeEnable": False, "ExposureTime": int(c["exposure_us"]),
                                      "AnalogueGain": float(c["gain"])})

    def read(self):
        return self.picam2.capture_array("main")

    def close(self):
        try:
            self.picam2.stop()
            self.picam2.close()
        except Exception:  # noqa
            pass


class SimCamera:
    """Synthetic camera: a dark-green ball rolling on the simulated tilting platform."""
    TRUE_ROT = 90.0   # how the simulated camera is rotated relative to the platform axes

    def __init__(self, cfg, platform):
        self.cfg = cfg
        self.platform = platform
        self.w, self.h = cfg.data["camera"]["capture"]
        self.ppc = 28.0 * self.w / 640.0     # capture px per cm
        self.pos = np.array([4.0, -3.0])      # cm, platform axes
        self.vel = np.zeros(2)
        self.last = time.perf_counter()
        self.next_t = self.last
        self.rng = np.random.default_rng(1)
        self.ball_bgr = (34, 96, 40)

    def apply_controls(self):
        pass

    def kick(self):
        ang = self.rng.uniform(0, 2 * math.pi)
        self.pos = np.array([math.cos(ang), math.sin(ang)]) * 4.5
        self.vel = np.zeros(2)

    def read(self):
        self.next_t += 1 / 60.0
        delay = self.next_t - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            self.next_t = time.perf_counter()
        now = time.perf_counter()
        dt = min(now - self.last, 0.05)
        self.last = now

        th, ph = math.radians(self.platform.theta), math.radians(self.platform.phi)
        acc = (5.0 / 7.0) * 981.0 * math.sin(th) * np.array([math.cos(ph), math.sin(ph)])
        self.vel += acc * dt
        self.vel *= max(0.0, 1.0 - 0.8 * dt)
        self.pos += self.vel * dt
        d = float(np.hypot(*self.pos))
        if d > 6.0:
            self.pos *= 6.0 / d
            self.vel[:] = 0.0

        cam = self.cfg.data["camera"]
        sx = self.w / float(cam["work"][0])
        cx, cy = cam["center"][0] * sx, cam["center"][1] * sx
        a = math.radians(-self.TRUE_ROT)
        ix = (self.pos[0] * math.cos(a) - self.pos[1] * math.sin(a)) * self.ppc
        iy = (self.pos[0] * math.sin(a) + self.pos[1] * math.cos(a)) * self.ppc

        frame = np.full((self.h, self.w, 3), (70, 66, 62), np.uint8)
        cv2.circle(frame, (int(cx), int(cy)), int(7.62 * self.ppc), (212, 208, 200), -1, cv2.LINE_AA)
        cv2.circle(frame, (int(cx), int(cy)), int(7.62 * self.ppc), (150, 145, 140), 3, cv2.LINE_AA)
        bx, by = int(cx + ix), int(cy + iy)
        rad = int(1.4 * self.ppc)
        cv2.circle(frame, (bx, by), rad, self.ball_bgr, -1, cv2.LINE_AA)
        hl = tuple(min(255, int(c * 0.7 + 255 * 0.3)) for c in self.ball_bgr)
        cv2.circle(frame, (bx - rad // 3, by - rad // 3), max(2, rad // 4), hl, -1, cv2.LINE_AA)
        noise = self.rng.normal(0, 3, frame.shape).astype(np.int16)
        return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Controller: capture -> track -> (PID) -> servos, in one thread
# ---------------------------------------------------------------------------
class Controller:
    def __init__(self, cfg, platform, camera, sim=False):
        self.cfg, self.platform, self.camera, self.sim = cfg, platform, camera, sim
        self.mode = "manual"                 # manual | auto | calibration | transition
        self.motion_lock = threading.Lock()  # held during blocking ramps
        self.tracker = Tracker()
        self.pid = PID()
        self.pid_info = {}
        self.pattern = TargetPattern()
        self.eff_target = tuple(cfg.data["camera"]["target"])   # target incl. any moving path offset
        self._pat_t = None
        self.running = False
        self.thread = None

        self.snap_lock = threading.Lock()
        self._frame = None
        self._det = None
        self._fid = 0

        self.ball_state = {"found": False, "x": 0.0, "y": 0.0, "r": 0.0, "vx": 0.0, "vy": 0.0, "lost_s": 0.0}
        self._bx = self._by = None
        self._last_t = None
        self._vx = self._vy = 0.0
        self._lost_since = None
        self.radius_hist = deque(maxlen=40)
        self.fps = 0.0
        self.proc_ms = 0.0

    # -- thread ----------------------------------------------------------
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        last_frame_t = time.perf_counter()
        while self.running:
            try:
                frame = self.camera.read()
                t0 = time.perf_counter()
                c = self.cfg.data
                cam, trk = c["camera"], c["tracking"]
                small = cv2.resize(frame, tuple(int(v) for v in cam["work"]), interpolation=cv2.INTER_AREA)
                det = self.tracker.detect(small, trk, cam["center"], cam["detection_radius"])
                self._update_ball(det, t0)
                self._update_target(t0)
                with self.snap_lock:
                    self._frame, self._det = frame, det
                    self._fid += 1
                if self.mode == "auto":
                    self._auto_step(t0)
                t1 = time.perf_counter()
                self.proc_ms = 0.9 * self.proc_ms + 0.1 * (t1 - t0) * 1000
                dtf = t1 - last_frame_t
                last_frame_t = t1
                if dtf > 0:
                    self.fps = 0.9 * self.fps + 0.1 / dtf
            except Exception as e:  # noqa
                log.exception("control loop error: %s", e)
                time.sleep(0.1)

    def snapshot(self):
        with self.snap_lock:
            return self._frame, self._det, self._fid

    # -- ball state ------------------------------------------------------
    def _update_ball(self, det, now):
        if det.found:
            if self._bx is not None and self._last_t is not None and self._lost_since is None:
                dt = max(now - self._last_t, 1e-3)
                self._vx = 0.7 * self._vx + 0.3 * (det.x - self._bx) / dt
                self._vy = 0.7 * self._vy + 0.3 * (det.y - self._by) / dt
            else:
                self._vx = self._vy = 0.0
            self._bx, self._by = det.x, det.y
            self._last_t = now
            self._lost_since = None
            self.radius_hist.append(det.r)
        else:
            if self._lost_since is None:
                self._lost_since = now
            if now - self._lost_since > 0.2:
                self._vx = self._vy = 0.0
            self._last_t = now
        lost_s = 0.0 if self._lost_since is None else now - self._lost_since
        self.ball_state = {"found": det.found, "x": self._bx, "y": self._by, "r": det.r if det.found else 0.0,
                           "vx": self._vx, "vy": self._vy, "lost_s": lost_s}

    # -- moving target ---------------------------------------------------
    def _update_target(self, now):
        cam = self.cfg.data["camera"]
        tx, ty = cam["target"]
        dt = 0.0 if self._pat_t is None else now - self._pat_t
        self._pat_t = now
        b = self.ball_state
        paused = (not b["found"]) and b["lost_s"] > 0.3
        off = self.pattern.update(dt, paused)
        if self.pattern.active:
            ppc = cam["px_per_cm"] or 14.0
            ex, ey = tx + off[0] * ppc, ty + off[1] * ppc
            cx, cy = cam["center"]
            lim = max(0.85 * cam["detection_radius"], math.hypot(tx - cx, ty - cy))
            d = math.hypot(ex - cx, ey - cy)
            if d > lim:
                ex, ey = cx + (ex - cx) * lim / d, cy + (ey - cy) * lim / d
            self.eff_target = (ex, ey)
        else:
            self.eff_target = (tx, ty)

    def safe_pattern_size(self):
        cam = self.cfg.data["camera"]
        tx, ty = cam["target"]
        cx, cy = cam["center"]
        room = 0.85 * cam["detection_radius"] - math.hypot(tx - cx, ty - cy)
        return max(0.0, room / (cam["px_per_cm"] or 14.0))

    # -- auto mode -------------------------------------------------------
    def _auto_step(self, now):
        c = self.cfg.data
        p, cam = c["pid"], c["camera"]
        hw = self.platform.h_work()
        b = self.ball_state
        if self._bx is None or b["lost_s"] > p["lost_timeout"]:
            self.pid.reset()
            th = self.platform.theta * 0.85
            self.platform.set_pose(0.0 if th < 0.05 else th, self.platform.phi, hw)
            self.pid_info = {}
            return
        tx, ty = self.eff_target
        norm = 200.0 / float(cam["work"][0])
        cap = min(p["tilt_limit"], self.platform.kin.max_theta(hw))
        theta, phi, info = self.pid.step((self._bx - tx, self._by - ty), now, p, norm, cap)
        self.platform.set_pose(theta, phi, hw)
        self.pid_info = info

    # -- mode / motion ---------------------------------------------------
    def set_mode(self, mode):
        if mode not in ("manual", "auto", "calibration"):
            return {"ok": False, "error": "unknown mode"}
        if not self.motion_lock.acquire(timeout=2.0):
            return {"ok": False, "error": "platform busy"}
        try:
            prev = self.mode
            if mode == prev:
                return {"ok": True}
            if mode != "auto":
                self.pattern.reset()
            if mode in ("auto", "calibration") or prev == "auto":
                self.mode = "transition"
                self.platform.ramp_to_pose(0.0, self.platform.phi, self.platform.h_work(), 0.6)
            if mode == "auto":
                self.pid.reset()
                self.pid_info = {}
            self.mode = mode
            return {"ok": True}
        finally:
            self.motion_lock.release()

    def auto_level(self):
        if not self.motion_lock.acquire(timeout=2.0):
            return {"ok": False, "error": "platform busy"}
        try:
            self.mode = "transition"
            self.pattern.reset()
            self.platform.ramp_to_pose(0.0, 0.0, self.platform.h_work(), 0.5)
            self.pid.reset()
            self.mode = "manual"
            return {"ok": True}
        finally:
            self.motion_lock.release()

    def manual_allowed(self):
        return self.mode in ("manual", "calibration") and not self.motion_lock.locked()

    def scale_from_ball(self):
        if len(self.radius_hist) < 10:
            return {"ok": False, "error": "Ball not tracked long enough - place it on the platform first"}
        r = float(np.median(self.radius_hist))
        dia = self.cfg.data["camera"]["ball_diameter_cm"]
        ppc = 2 * r / dia
        self.cfg.patch({"camera": {"px_per_cm": ppc}})
        return {"ok": True, "px_per_cm": self.cfg.data["camera"]["px_per_cm"], "radius_px": r}

    def pick_color(self, x, y, hue_margin=6.0, sv_margin=0.2, set_size=True):
        """Learn the ball colour from a click: grow the clicked colour region, then set the HSV
        range from robust percentiles of the pixels inside it (hue handled circularly)."""
        frame, _, _ = self.snapshot()
        if frame is None:
            return {"ok": False, "error": "No camera frame yet"}
        with self.cfg.lock:
            cam = dict(self.cfg.data["camera"])
            trk = dict(self.cfg.data["tracking"])
        small = cv2.resize(frame, tuple(int(v) for v in cam["work"]), interpolation=cv2.INTER_AREA)
        k = int(trk["blur"]) | 1
        img = cv2.GaussianBlur(small, (k, k), 0) if k > 1 else small
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H, W = hsv.shape[:2]
        xi = int(min(max(round(x), 0), W - 1))
        yi = int(min(max(round(y), 0), H - 1))
        y0, y1, x0, x1 = max(0, yi - 3), min(H, yi + 4), max(0, xi - 3), min(W, xi + 4)

        patch = hsv[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
        ang = patch[:, 0] * (2 * np.pi / 180.0)
        h_med = (math.degrees(math.atan2(float(np.sin(ang).mean()), float(np.cos(ang).mean()))) % 360.0) / 2.0
        s_med, v_med = float(np.median(patch[:, 1])), float(np.median(patch[:, 2]))
        low_sat = s_med < 45          # white / grey / black ball: hue is meaningless

        hh = hsv[:, :, 0].astype(np.float32)
        S = hsv[:, :, 1].astype(np.float32)
        V = hsv[:, :, 2].astype(np.float32)
        dh = np.abs((hh - h_med + 90.0) % 180.0 - 90.0)
        if low_sat:
            cand = (S <= s_med + 45) & (np.abs(V - v_med) <= 60)
        else:
            cand = (dh <= 22) & (S >= 0.5 * s_med) & (V >= 0.4 * v_med)
        cand = cv2.morphologyEx(cand.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        _, lab = cv2.connectedComponents(cand)
        lbl = int(lab[yi, xi])
        if lbl == 0:
            sub = lab[y0:y1, x0:x1]
            nz = sub[sub > 0]
            if nz.size == 0:
                return {"ok": False, "error": "No clear colour at that spot - click in the middle of the ball"}
            lbl = int(np.bincount(nz.ravel()).argmax())
        comp = lab == lbl
        area = int(comp.sum())
        frac = area / float(W * H)
        if area < 20:
            return {"ok": False, "error": "The blob at that spot is too small - click in the middle of the ball"}
        if frac > 0.25:
            return {"ok": False, "error": f"That colour covers {frac * 100:.0f}% of the picture, so it is too close to the "
                                          "platform/background. Click the ball's most vivid area, or use a more "
                                          "contrasting ball."}

        er = cv2.erode(comp.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        sel = er if int(er.sum()) >= 15 else comp
        hs = hsv[sel].astype(np.float32)
        d = (hs[:, 0] - h_med + 90.0) % 180.0 - 90.0
        d_lo, d_hi = np.percentile(d, [2, 98])
        s_lo, s_hi = np.percentile(hs[:, 1], [2, 98])
        v_lo, v_hi = np.percentile(hs[:, 2], [2, 98])

        if low_sat:
            h_min, h_max = 0, 179
            s_min, s_max = 0, min(255, s_hi * (1 + sv_margin) + 15)
        else:
            half_lo = float(np.clip(hue_margin - d_lo, hue_margin, 30.0))
            half_hi = float(np.clip(hue_margin + d_hi, hue_margin, 30.0))
            h_min = int(round(h_med - half_lo)) % 180
            h_max = int(round(h_med + half_hi)) % 180
            s_min, s_max = max(0.0, s_lo * (1 - sv_margin) - 5), 255
        v_min = max(0.0, v_lo * (1 - sv_margin) - 5)
        v_max = min(255.0, v_hi * (1 + sv_margin) + 10)

        r = math.sqrt(area / math.pi)
        new = {"h_min": h_min, "h_max": h_max, "s_min": int(s_min), "s_max": int(s_max),
               "v_min": int(v_min), "v_max": int(v_max)}
        if set_size:
            new["min_radius"] = round(max(3.0, 0.55 * r), 1)
            new["max_radius"] = round(max(new["min_radius"] + 2, 1.8 * r), 1)
        self.cfg.patch({"tracking": new})

        with self.cfg.lock:
            trk2 = dict(self.cfg.data["tracking"])
        det = self.tracker.detect(small, trk2, cam["center"], cam["detection_radius"])
        return {"ok": True, "tracking": trk2, "found": det.found, "radius_px": r,
                "seed_hsv": [h_med, s_med, v_med], "low_sat": low_sat, "area_pct": frac * 100}

    # -- state for the UI --------------------------------------------------
    def get_state(self):
        c = self.cfg.data
        cam = c["camera"]
        ppc = cam["px_per_cm"] or 14.0
        b = self.ball_state
        tx, ty = self.eff_target
        if b["x"] is not None:
            ex, ey = b["x"] - tx, b["y"] - ty
        else:
            ex = ey = 0.0
        pl = self.platform
        angles = [a if a is not None else 0.0 for a in pl.angles]
        return {
            "ok": True, "sim": self.sim, "mode": self.mode,
            "fps": self.fps, "proc_ms": self.proc_ms,
            "ball": {
                "found": b["found"], "has_pos": b["x"] is not None, "lost_s": b["lost_s"],
                "x": b["x"], "y": b["y"], "r": b["r"],
                "ex_px": ex, "ey_px": ey, "dist_px": math.hypot(ex, ey),
                "ex_cm": ex / ppc, "ey_cm": ey / ppc, "dist_cm": math.hypot(ex, ey) / ppc,
                "vx_cm": b["vx"] / ppc, "vy_cm": b["vy"] / ppc, "speed_cm": math.hypot(b["vx"], b["vy"]) / ppc,
                "vx_px": b["vx"], "vy_px": b["vy"],
            },
            "pose": {"theta": pl.theta, "phi": pl.phi, "h": pl.h,
                     "theta_max": pl.kin.max_theta(pl.h_work()), "h_work": pl.h_work(),
                     "h_min": pl.h_range[0], "h_max": pl.h_range[1]},
            "servo": {"angles": angles, "clamped": pl.clamped, "limits": pl.limits()},
            "pid": self.pid_info,
            "target": [tx, ty],
            "pattern": dict(self.pattern.state(), safe_size=self.safe_pattern_size()),
        }

    def derived(self):
        pl = self.platform
        return {"h_range": list(pl.h_range), "theta_max": pl.kin.max_theta(pl.h_work()),
                "kin_h_range": [pl.kin.min_h, pl.kin.max_h], "sim": self.sim}

    def shutdown(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        try:
            self.platform.ramp_to_pose(0.0, 0.0, self.platform.h_work(), 0.4)
        except Exception:  # noqa
            pass
        self.camera.close()


# ---------------------------------------------------------------------------
# Video overlay
# ---------------------------------------------------------------------------
C_CENTER = (255, 190, 70)
C_TARGET = (0, 160, 255)
C_BALL = (70, 70, 255)
C_ERR = (0, 230, 255)
STATUS_COLORS = {"ok": (0, 255, 0), "size": (150, 150, 150), "shape": (0, 215, 255), "outside": (255, 120, 160)}


def _cross(img, x, y, n, col, th):
    x, y = int(x), int(y)
    cv2.line(img, (x - n, y), (x + n, y), col, th, cv2.LINE_AA)
    cv2.line(img, (x, y - n), (x, y + n), col, th, cv2.LINE_AA)


def _text(img, s, org, col, scale=0.5):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


def render_frame(frame, det, ctrl, view):
    if view == "raw":
        return frame
    cam = ctrl.cfg.data["camera"]
    h, w = frame.shape[:2]
    sc = w / float(cam["work"][0])
    if view == "mask" and det is not None and det.mask is not None:
        m = cv2.resize(det.mask, (w, h), interpolation=cv2.INTER_NEAREST)
        img = cv2.convertScaleAbs(frame, alpha=0.25)
        img[m > 0] = (110, 255, 140)
    else:
        img = frame.copy()

    cx, cy = cam["center"]
    tx, ty = ctrl.eff_target
    cv2.circle(img, (int(cx * sc), int(cy * sc)), int(cam["detection_radius"] * sc), C_CENTER, 1, cv2.LINE_AA)
    _cross(img, cx * sc, cy * sc, 8, C_CENTER, 1)
    if ctrl.pattern.active:
        bx0, by0 = cam["target"]
        ppc = cam["px_per_cm"] or 14.0
        pts = np.array([[(bx0 + x * ppc) * sc, (by0 + y * ppc) * sc] for x, y in ctrl.pattern.preview()], np.int32)
        cv2.polylines(img, [pts], False, (0, 110, 170), 1, cv2.LINE_AA)
        _cross(img, bx0 * sc, by0 * sc, 5, (0, 110, 170), 1)
    _cross(img, tx * sc, ty * sc, 13, C_TARGET, 2)
    cv2.circle(img, (int(tx * sc), int(ty * sc)), int(5 * sc), C_TARGET, 1, cv2.LINE_AA)

    if det is not None:
        for c, status in det.candidates:
            pts = (c.astype(np.float32) * sc).astype(np.int32)
            cv2.polylines(img, [pts], True, STATUS_COLORS[status], 2 if status == "ok" else 1, cv2.LINE_AA)

    b = ctrl.ball_state
    if b["x"] is not None:
        bx, by = b["x"] * sc, b["y"] * sc
        col = C_BALL if b["found"] else (120, 120, 120)
        cv2.line(img, (int(bx), int(by)), (int(tx * sc), int(ty * sc)), C_ERR, 1, cv2.LINE_AA)
        cv2.circle(img, (int(bx), int(by)), max(4, int(b["r"] * sc)) if b["found"] else 10, col, 2, cv2.LINE_AA)
        _cross(img, bx, by, 6, col, 2)
    label = {"manual": "MANUAL", "auto": "AUTO", "calibration": "CALIBRATION", "transition": "MOVING..."}[ctrl.mode]
    if ctrl.pattern.active:
        label += f" - {ctrl.pattern.shape.upper()} PATH"
    _text(img, f"{label}   {ctrl.fps:0.0f} fps", (10, 22), (255, 255, 255))
    if not b["found"]:
        _text(img, "BALL NOT FOUND", (10, 46), (80, 80, 255), 0.6)
    return img


# ---------------------------------------------------------------------------
# Web app
# ---------------------------------------------------------------------------
app = Flask(__name__)
presets = None
cfg = None
platform = None
ctrl = None


def _config_payload():
    return {"ok": True, "config": cfg.data, "derived": ctrl.derived()}


@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    if request.method == "POST":
        patch = request.get_json(force=True, silent=True) or {}
        changed = cfg.patch(patch)
        err = None
        if any(k.startswith(("geometry.", "servo.")) for k in changed):
            err = platform.rebuild()
        if changed & {"camera.exposure_auto", "camera.exposure_us", "camera.gain", "camera.awb_auto"}:
            ctrl.camera.apply_controls()
        if ctrl.mode == "calibration" and any(k in changed for k in ("servo.offsets", "platform.h_work")):
            if ctrl.manual_allowed():
                platform.set_pose(0.0, 0.0, platform.h_work())
        out = _config_payload()
        if err:
            out["warning"] = err
        return jsonify(out)
    return jsonify(_config_payload())


@app.route("/api/state")
def api_state():
    return jsonify(ctrl.get_state())


@app.route("/api/mode", methods=["POST"])
def api_mode():
    return jsonify(ctrl.set_mode((request.get_json(force=True, silent=True) or {}).get("mode")))


@app.route("/api/autolevel", methods=["POST"])
def api_autolevel():
    return jsonify(ctrl.auto_level())


@app.route("/api/servo", methods=["POST"])
def api_servo():
    d = request.get_json(force=True, silent=True) or {}
    if not ctrl.manual_allowed():
        return jsonify({"ok": False, "error": "Servo sliders only work in Manual or Calibration mode"})
    try:
        platform.set_servo(int(d["index"]), float(d["angle"]))
    except (KeyError, ValueError, IndexError):
        return jsonify({"ok": False, "error": "bad request"})
    return jsonify({"ok": True})


@app.route("/api/pose", methods=["POST"])
def api_pose():
    d = request.get_json(force=True, silent=True) or {}
    if not ctrl.manual_allowed():
        return jsonify({"ok": False, "error": "Pose sliders only work in Manual or Calibration mode"})
    try:
        ok = platform.set_pose(float(d.get("theta", platform.theta)), float(d.get("phi", platform.phi)),
                               float(d.get("h", platform.h)))
    except ValueError:
        return jsonify({"ok": False, "error": "bad request"})
    return jsonify({"ok": ok, "error": None if ok else "pose not reachable"})


@app.route("/api/limit", methods=["POST"])
def api_limit():
    """Set a hard limit and move that servo to it so the user can see where it binds."""
    d = request.get_json(force=True, silent=True) or {}
    if ctrl.mode != "calibration" or ctrl.motion_lock.locked():
        return jsonify({"ok": False, "error": "Limits can only be changed in Calibration mode"})
    try:
        i, which, value = int(d["index"]), d["which"], float(d["value"])
        key = "limit_down" if which == "down" else "limit_up"
        lst = list(cfg.data["servo"][key])
        lst[i] = value
        cfg.patch({"servo": {key: lst}})
        platform.rebuild()
        platform.set_servo_free(i, cfg.data["servo"][key][i])
    except (KeyError, ValueError, IndexError):
        return jsonify({"ok": False, "error": "bad request"})
    return jsonify({"ok": True, "derived": ctrl.derived()})


@app.route("/api/save", methods=["POST"])
def api_save():
    try:
        cfg.save()
        return jsonify({"ok": True, "path": cfg.path})
    except Exception as e:  # noqa
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/pid_reset", methods=["POST"])
def api_pid_reset():
    ctrl.pid.reset()
    return jsonify({"ok": True})


@app.route("/api/scale_from_ball", methods=["POST"])
def api_scale():
    out = ctrl.scale_from_ball()
    if out.get("ok"):
        out["config"] = cfg.data
    return jsonify(out)


def _presets_payload():
    return {"ok": True, "presets": presets.listing(), "keys": PRESET_KEYS}


@app.route("/api/presets")
def api_presets():
    return jsonify(_presets_payload())


@app.route("/api/presets/save", methods=["POST"])
def api_presets_save():
    d = request.get_json(force=True, silent=True) or {}
    ok, err = presets.save(d.get("name"), preset_values(cfg), bool(d.get("overwrite")))
    out = _presets_payload()
    out["ok"] = ok
    if ok:
        out["name"] = PresetStore.clean(d.get("name"))
    else:
        out["exists"] = err == "exists"
        out["error"] = "A preset with that name already exists" if err == "exists" else err
    return jsonify(out)


@app.route("/api/presets/load", methods=["POST"])
def api_presets_load():
    d = request.get_json(force=True, silent=True) or {}
    values = presets.get(d.get("name"))
    if values is None:
        return jsonify({"ok": False, "error": "No such preset"})
    cfg.patch(values)
    ctrl.pid.reset()
    ctrl.pid_info = {}
    out = _config_payload()
    out["name"] = PresetStore.clean(d.get("name"))
    return jsonify(out)


@app.route("/api/presets/delete", methods=["POST"])
def api_presets_delete():
    d = request.get_json(force=True, silent=True) or {}
    ok, err = presets.delete(d.get("name"))
    out = _presets_payload()
    out["ok"] = ok
    if not ok:
        out["error"] = err
    return jsonify(out)


@app.route("/api/presets/rename", methods=["POST"])
def api_presets_rename():
    d = request.get_json(force=True, silent=True) or {}
    ok, err = presets.rename(d.get("name"), d.get("new_name"))
    out = _presets_payload()
    out["ok"] = ok
    if ok:
        out["name"] = PresetStore.clean(d.get("new_name"))
    else:
        out["error"] = err
    return jsonify(out)


@app.route("/api/pattern", methods=["POST"])
def api_pattern():
    d = request.get_json(force=True, silent=True) or {}
    action = d.get("action", "config")
    try:
        if action == "start":
            if ctrl.mode != "auto":
                return jsonify({"ok": False, "error": "Switch to Auto mode first - the path runs while the PID is balancing"})
            ctrl.pattern.configure(d)
            ctrl.pattern.start()
        elif action == "stop":
            ctrl.pattern.stop()
        else:
            ctrl.pattern.configure(d)
    except (ValueError, TypeError):
        return jsonify({"ok": False, "error": "bad request"})
    return jsonify({"ok": True, "pattern": ctrl.pattern.state()})


@app.route("/api/pick_color", methods=["POST"])
def api_pick_color():
    d = request.get_json(force=True, silent=True) or {}
    try:
        out = ctrl.pick_color(float(d["x"]), float(d["y"]), float(d.get("hue_margin", 6.0)),
                              float(d.get("sv_margin", 0.2)), bool(d.get("set_size", True)))
    except (KeyError, ValueError):
        return jsonify({"ok": False, "error": "bad request"})
    return jsonify(out)


@app.route("/api/sim/color", methods=["POST"])
def api_sim_color():
    if not ctrl.sim:
        return jsonify({"ok": False, "error": "not in simulation"})
    d = request.get_json(force=True, silent=True) or {}
    ctrl.camera.ball_bgr = tuple(int(min(max(d.get(k, 0), 0), 255)) for k in ("b", "g", "r"))
    return jsonify({"ok": True})


@app.route("/api/sim/kick", methods=["POST"])
def api_kick():
    if ctrl.sim:
        ctrl.camera.kick()
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "not in simulation"})


@app.route("/video_feed")
def video_feed():
    view = request.args.get("view", "overlay")

    def gen():
        last = -1
        while True:
            frame, det, fid = ctrl.snapshot()
            if frame is None or fid == last:
                time.sleep(0.005)
                continue
            last = fid
            img = render_frame(frame, det, ctrl, view)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 72])
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            time.sleep(0.03)

    resp = Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")
    resp.headers["Cache-Control"] = "no-store"
    return resp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:  # noqa
        return "localhost"


def main():
    global cfg, platform, ctrl, presets
    ap = argparse.ArgumentParser(description="Ball balancing robot controller + web UI")
    ap.add_argument("--sim", action="store_true", help="simulated camera and servos (no hardware)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    cfg = Config(args.config)
    presets = PresetStore(os.path.join(os.path.dirname(os.path.abspath(args.config)), "balancer_presets.json"))
    driver = MockDriver() if args.sim else ServoDriver(cfg.data["servo"])
    platform = Platform(cfg, driver)
    camera = SimCamera(cfg, platform) if args.sim else PiCamera(cfg)
    ctrl = Controller(cfg, platform, camera, sim=args.sim)
    ctrl.start()

    lo, hi = platform.h_range
    log.info("Platform height range (level, inside servo limits): %.2f - %.2f cm", lo, hi)
    log.info("Open  http://%s:%d", local_ip(), args.port)

    def _stop(sig, frm):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _stop)
    try:
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        log.info("Shutting down - levelling platform")
        ctrl.shutdown()


# ---------------------------------------------------------------------------
# Embedded web UI
# ---------------------------------------------------------------------------
INDEX_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ball Balancer</title>
<style>
:root{--bg:#090d13;--panel:#111822;--panel2:#161f2c;--line:#223043;--text:#e8eef6;--muted:#8393a8;
--accent:#3ee6a8;--blue:#5aa9ff;--warn:#ffb547;--danger:#ff5d6e;--r:14px}
*{box-sizing:border-box}
html,body{margin:0}
body{background:radial-gradient(1100px 600px at 15% -10%,#13222f 0%,var(--bg) 60%) fixed;color:var(--text);
font:14px/1.4 Inter,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;min-height:100vh}
button,input,select{font:inherit;color:inherit}
.app{display:grid;grid-template-columns:minmax(0,1fr) 450px;gap:16px;padding:16px;max-width:1700px;margin:0 auto}
header.top{grid-column:1/-1;display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.brand{display:flex;align-items:center;gap:10px;font-weight:650;font-size:17px;letter-spacing:.2px}
.logo{width:30px;height:30px;border-radius:9px;background:linear-gradient(135deg,var(--accent),#1aa57a);display:grid;place-items:center;color:#04130d;font-weight:800}
.pills{display:flex;gap:8px;flex-wrap:wrap;margin-left:6px}
.pill{display:flex;align-items:center;gap:7px;padding:5px 11px;border-radius:99px;background:var(--panel);border:1px solid var(--line);color:var(--muted);font-size:12.5px}
.pill b{color:var(--text);font-weight:600}
.dot{width:8px;height:8px;border-radius:50%;background:var(--danger)}
.dot.ok{background:var(--accent);box-shadow:0 0 8px var(--accent)}
.dot.warn{background:var(--warn)}
.spacer{flex:1}
.btn{border:1px solid var(--line);background:var(--panel2);padding:8px 14px;border-radius:10px;cursor:pointer;transition:.15s;font-weight:550}
.btn:hover{border-color:#34506f;background:#1b2636}
.btn.primary{background:var(--accent);border-color:var(--accent);color:#04130d}
.btn.primary:hover{filter:brightness(1.08)}
.btn.warn{background:rgba(255,181,71,.12);border-color:rgba(255,181,71,.55);color:var(--warn)}
.btn.warn:hover{background:rgba(255,181,71,.2)}
.btn.sm{padding:5px 10px;font-size:12.5px;border-radius:8px}
.btn.on{background:rgba(62,230,168,.16);border-color:var(--accent);color:var(--accent)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:var(--r);box-shadow:0 10px 30px rgba(0,0,0,.25)}
.left{display:flex;flex-direction:column;gap:14px;min-width:0}
.video-card{padding:12px}
.vbar{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:10px}
.seg{display:inline-flex;background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:3px;gap:2px}
.seg button{background:none;border:0;padding:6px 13px;border-radius:7px;cursor:pointer;color:var(--muted);font-weight:550}
.seg button.on{background:var(--panel2);color:var(--text);box-shadow:0 1px 0 rgba(255,255,255,.04) inset}
.seg.big{display:flex;width:100%}.seg.big button{flex:1;padding:9px}
.seg.big button.on{background:var(--accent);color:#04130d}
.legend{display:flex;gap:12px;flex-wrap:wrap;margin-left:auto;color:var(--muted);font-size:12px}
.legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;vertical-align:-1px}
:root{--voff:300px}
@media(max-width:1300px){:root{--voff:382px}}
#vwrap{position:relative;aspect-ratio:4/3;width:min(100%,calc((100vh - var(--voff)) * 1.3333 * var(--vs,1)));min-width:min(300px,100%);margin:0 auto;background:#000;border-radius:10px;overflow:hidden;border:1px solid var(--line)}
#video{width:100%;height:100%;display:block;object-fit:fill;user-select:none;-webkit-user-drag:none}
#vover{position:absolute;inset:0;touch-action:none}
#vover.active{cursor:crosshair;outline:2px dashed rgba(62,230,168,.7);outline-offset:-3px}
#toolhint{position:absolute;left:10px;bottom:10px;background:rgba(9,13,19,.85);padding:5px 10px;border-radius:8px;font-size:12.5px;color:var(--accent);pointer-events:none}
#toolhint:empty{display:none}
.tiles{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px}
@media(max-width:1300px){.tiles{grid-template-columns:repeat(3,minmax(0,1fr))}}
.tile{padding:10px 12px}
.tile .l{color:var(--muted);font-size:11.5px;text-transform:uppercase;letter-spacing:.7px}
.tile .v{font-size:17px;font-weight:650;margin-top:3px;font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tile .s{color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bad{color:var(--danger)!important}.good{color:var(--accent)!important}.amber{color:var(--warn)!important}
aside.right{padding:14px;align-self:start;position:sticky;top:16px;max-height:calc(100vh - 32px);overflow:auto}
.page{display:none;margin-top:14px}.page.on{display:block}
details.sec{border:1px solid var(--line);background:var(--panel2);border-radius:12px;margin-bottom:12px}
details.sec>summary{cursor:pointer;list-style:none;padding:11px 14px;font-weight:620;display:flex;justify-content:space-between;align-items:center}
details.sec>summary::-webkit-details-marker{display:none}
details.sec>summary:after{content:"▾";color:var(--muted);transition:.15s}
details.sec:not([open])>summary:after{transform:rotate(-90deg)}
.body{padding:2px 14px 14px}
.note{color:var(--muted);font-size:12.2px;margin:2px 0 10px}
.row{margin:11px 0}
.rl{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;color:var(--muted);font-size:13px}
.rl span:first-child{color:var(--text)}
.rv{display:flex;align-items:center;gap:5px}.rv i{font-style:normal;color:var(--muted);font-size:12px;min-width:12px}
input[type=number],select{background:var(--bg);border:1px solid var(--line);border-radius:7px;padding:5px 8px;width:118px;text-align:right;font-variant-numeric:tabular-nums;-moz-appearance:textfield;appearance:textfield}
input[type=number]::-webkit-inner-spin-button,input[type=number]::-webkit-outer-spin-button{-webkit-appearance:none;margin:0}
select{width:auto;text-align:left}
input[type=number]:focus,select:focus{outline:1px solid var(--accent);border-color:var(--accent)}
input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:5px;border-radius:5px;outline:none;margin:6px 0;
background:linear-gradient(to right,var(--accent) var(--p,50%),var(--line) var(--p,50%))}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:17px;height:17px;border-radius:50%;background:var(--text);border:3px solid var(--accent);cursor:pointer}
input[type=range]::-moz-range-thumb{width:12px;height:12px;border-radius:50%;background:var(--text);border:3px solid var(--accent);cursor:pointer}
.row.clamped input[type=range]{background:linear-gradient(to right,var(--danger) var(--p,50%),var(--line) var(--p,50%))}
.cap{display:flex;justify-content:space-between;color:var(--muted);font-size:11.5px}
.kv{display:grid;grid-template-columns:1fr auto;gap:5px 12px;font-variant-numeric:tabular-nums}
.kv div:nth-child(odd){color:var(--muted)}.kv div:nth-child(even){text-align:right;font-weight:560}
.toggle{display:flex;justify-content:space-between;align-items:center;margin:12px 0}
.toggle input{width:38px;height:22px;-webkit-appearance:none;appearance:none;background:var(--line);border-radius:22px;position:relative;cursor:pointer;transition:.15s}
.toggle input:after{content:"";position:absolute;top:3px;left:3px;width:16px;height:16px;border-radius:50%;background:var(--text);transition:.15s}
.toggle input:checked{background:var(--accent)}.toggle input:checked:after{left:19px;background:#04130d}
.btns{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 2px}
#hueBar{height:10px;border-radius:6px;margin:4px 0 12px;border:1px solid var(--line)}
body[data-mode=auto] .manual-only{opacity:.4;pointer-events:none}
#toast{position:fixed;right:18px;bottom:18px;display:flex;flex-direction:column;gap:8px;z-index:9}
.t{background:var(--panel2);border:1px solid var(--line);border-left:3px solid var(--accent);padding:10px 14px;border-radius:10px;box-shadow:0 8px 24px rgba(0,0,0,.4);max-width:340px}
.t.err{border-left-color:var(--danger)}
.presetbar{margin-top:12px;padding:10px 12px;border:1px solid var(--line);background:var(--panel2);border-radius:12px}
.presetbar .plabel{color:var(--muted);font-size:11.5px;text-transform:uppercase;letter-spacing:.7px;margin-bottom:7px;display:flex;justify-content:space-between;align-items:center;gap:8px}
.presetbar select{width:100%;text-align:left;padding:7px 8px}
.presetbar .btns{margin:8px 0 0}
.badge{font-size:11.5px;padding:2px 9px;border-radius:99px;border:1px solid var(--line);color:var(--muted);white-space:nowrap;text-transform:none;letter-spacing:0}
.badge.mod{color:var(--warn);border-color:rgba(255,181,71,.5);background:rgba(255,181,71,.1)}
.badge.ok{color:var(--accent);border-color:rgba(62,230,168,.45);background:rgba(62,230,168,.08)}
.btn:disabled{opacity:.4;cursor:not-allowed}
.btn.danger{background:rgba(255,93,110,.12);border-color:rgba(255,93,110,.5);color:var(--danger)}
.modalbg{position:fixed;inset:0;background:rgba(3,6,10,.62);display:none;align-items:center;justify-content:center;z-index:20;backdrop-filter:blur(3px)}
.modalbg.on{display:flex}
.modal{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px;width:min(380px,92vw);box-shadow:0 20px 60px rgba(0,0,0,.5)}
.modal h3{margin:0 0 6px;font-size:16px}
.modal p{margin:0 0 14px;color:var(--muted);font-size:13px}
.modal input[type=text]{width:100%;background:var(--bg);border:1px solid var(--line);border-radius:9px;padding:9px 11px;margin-bottom:14px}
.modal input[type=text]:focus{outline:1px solid var(--accent);border-color:var(--accent)}
.modal .mb{display:flex;justify-content:flex-end;gap:8px}
.shapes{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:8px 0 4px}
.shapes button{display:flex;flex-direction:column;align-items:center;gap:4px;padding:9px 4px;border:1px solid var(--line);background:var(--bg);border-radius:10px;cursor:pointer;color:var(--muted);font-size:12px}
.shapes button svg{width:26px;height:26px;stroke:currentColor;fill:none;stroke-width:1.8;stroke-linejoin:round;stroke-linecap:round}
.shapes button.on{color:var(--accent);border-color:var(--accent);background:rgba(62,230,168,.1)}
.pstat{display:flex;align-items:center;gap:8px;margin:12px 0 2px;color:var(--muted);font-size:13px}
@media(max-width:1020px){.app{grid-template-columns:1fr}aside.right{position:static;max-height:none}.tiles{grid-template-columns:repeat(2,1fr)}}
</style></head><body data-mode="manual">
<div class="app">
<header class="top">
  <div class="brand"><div class="logo">B</div>Ball Balancer</div>
  <div class="pills">
    <div class="pill"><span class="dot" id="dConn"></span><span id="pConn">connecting</span></div>
    <div class="pill">Mode <b id="pMode">–</b></div>
    <div class="pill"><span class="dot" id="dBall"></span><span id="pBall">ball</span></div>
    <div class="pill" id="pSim" style="display:none"><b>SIMULATION</b></div>
  </div>
  <div class="spacer"></div>
  <button class="btn sm" id="btnKick" style="display:none">Nudge ball (sim)</button>
  <button class="btn warn" id="btnLevel" title="Stop tilting and go to the neutral level position at the work height">⏏ Auto Level</button>
  <button class="btn primary" id="btnSave">Save settings</button>
</header>

<main class="left">
  <section class="card video-card">
    <div class="vbar">
      <div class="seg" id="viewSeg"><button data-v="overlay" class="on">Overlay</button><button data-v="mask">Mask</button><button data-v="raw">Raw</button></div>
      <div class="seg" id="sizeSeg" title="Preview size"><button data-z="S">S</button><button data-z="M">M</button><button data-z="L" class="on">L</button></div>
      <div class="legend">
        <span><i style="background:#46beff"></i>center / detect area</span>
        <span><i style="background:#ffa000"></i>target</span>
        <span><i style="background:#ff4646"></i>ball</span>
        <span><i style="background:#0f0"></i>accepted contour</span>
        <span><i style="background:#999"></i>rejected: size</span>
        <span><i style="background:#ffd700"></i>shape</span>
      </div>
    </div>
    <div id="vwrap"><img id="video" alt="camera"><div id="vover"></div><div id="toolhint"></div></div>
  </section>
  <section class="tiles" id="tiles"></section>
</main>

<aside class="right card">
  <div class="seg big" id="modeSeg"><button data-m="manual" class="on">Manual</button><button data-m="auto">Auto</button><button data-m="calibration">Calibration</button></div>
  <div class="presetbar" id="presetBar">
    <div class="plabel"><span>Tuning preset · PID + ball</span><span class="badge" id="pBadge">none loaded</span></div>
    <select id="tunSel"></select>
    <div class="btns">
      <button class="btn sm primary" id="pSave" title="Overwrite the loaded preset with the current values">Save</button>
      <button class="btn sm" id="pSaveAs">Save as new…</button>
      <button class="btn sm" id="pRename">Rename</button>
      <button class="btn sm danger" id="pDelete">Delete</button>
    </div>
  </div>
  <div id="pages">
    <div class="page on" data-page="manual"></div>
    <div class="page" data-page="auto"></div>
    <div class="page" data-page="calibration"></div>
  </div>
</aside>
</div>
<div id="toast"></div>
<div class="modalbg" id="modalBg"><div class="modal"></div></div>

<script>
(()=>{'use strict';
const $=(s,e=document)=>e.querySelector(s);
const el=(t,c,h)=>{const n=document.createElement(t);if(c)n.className=c;if(h!=null)n.innerHTML=h;return n;};
const clamp=(v,a,b)=>Math.min(b,Math.max(a,v));
const fmt=(v,d=1)=>(v==null||isNaN(v))?'–':Number(v).toFixed(d);
const PRESETS={
 'Dark green':{h_min:35,h_max:90,s_min:60,s_max:255,v_min:20,v_max:200},
 'Green':{h_min:38,h_max:85,s_min:80,s_max:255,v_min:60,v_max:255},
 'Red (wraps)':{h_min:170,h_max:8,s_min:120,s_max:255,v_min:70,v_max:255},
 'Orange':{h_min:6,h_max:20,s_min:130,s_max:255,v_min:90,v_max:255},
 'Yellow':{h_min:21,h_max:35,s_min:100,s_max:255,v_min:100,v_max:255},
 'Blue':{h_min:98,h_max:128,s_min:100,s_max:255,v_min:50,v_max:255},
 'Purple':{h_min:129,h_max:150,s_min:60,s_max:255,v_min:40,v_max:255},
 'Pink / magenta':{h_min:150,h_max:172,s_min:70,s_max:255,v_min:90,v_max:255},
 'White':{h_min:0,h_max:179,s_min:0,s_max:40,v_min:190,v_max:255},
 'Black':{h_min:0,h_max:179,s_min:0,s_max:255,v_min:0,v_max:55}
};
const PICK={hue:8,sv:25,size:true};
const PAT={shape:'circle',size:2.5,speed:3,direction:'cw',rot:0};
let patUI=null,PATSYNC=false;
let TUN=[],TUNKEYS={},ACTIVE=null,VZ='L';
try{ACTIVE=localStorage.getItem('bb_active_preset')||null;VZ=localStorage.getItem('bb_vz')||'L';}catch(e){}

let CFG=null,DER={},ST=null,TOOL=null,MODE='manual',modeLock=0,connOK=false;
const bindings=[],dragging={};
const getPath=(o,p)=>p.split('.').reduce((a,k)=>a==null?a:a[k],o);
const setPath=(o,p,v)=>{const k=p.split('.');const l=k.pop();k.reduce((a,x)=>a[x],o)[l]=v;};
const deepMerge=(a,b)=>{for(const k in b){if(b[k]&&typeof b[k]==='object'&&!Array.isArray(b[k])){a[k]=a[k]||{};deepMerge(a[k],b[k]);}else a[k]=b[k];}return a;};
function patchFor(path){
  const parts=path.split('.');
  if(/^\d+$/.test(parts[parts.length-1]))parts.pop();
  const val=JSON.parse(JSON.stringify(getPath(CFG,parts.join('.'))));
  const patch={};let cur=patch;
  parts.forEach((k,i)=>{if(i===parts.length-1)cur[k]=val;else{cur[k]={};cur=cur[k];}});
  return patch;
}

/* ---------- networking ---------- */
async function api(url,body){
  try{
    const r=await fetch(url,{method:body===undefined?'GET':'POST',headers:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});
    return await r.json();
  }catch(e){return {ok:false,error:String(e)};}
}
function toast(msg,err){const t=el('div','t'+(err?' err':''),msg);$('#toast').appendChild(t);setTimeout(()=>t.remove(),err?5000:2600);}
const handle=r=>{if(r&&r.ok===false&&r.error)toast(r.error,true);return r;};
const pending={config:null,servo:{},pose:null,limit:{},pattern:null};let timer=null;
const schedule=()=>{if(!timer)timer=setTimeout(flush,45);};
function flush(){
  timer=null;
  if(pending.config){const p=pending.config;pending.config=null;api('/api/config',p).then(r=>{if(r&&r.derived){DER=r.derived;updateRanges();}if(r&&r.warning)toast(r.warning,true);});}
  for(const i in pending.servo){const a=pending.servo[i];delete pending.servo[i];api('/api/servo',{index:+i,angle:a}).then(handle);}
  if(pending.pose){const p=pending.pose;pending.pose=null;api('/api/pose',p).then(handle);}
  if(pending.pattern){const p=pending.pattern;pending.pattern=null;api('/api/pattern',{action:'config',...p}).then(handle);}
  for(const k in pending.limit){const v=pending.limit[k];delete pending.limit[k];api('/api/limit',v).then(r=>{handle(r);if(r&&r.derived){DER=r.derived;updateRanges();}});}
}
function queuePatch(path){pending.config=deepMerge(pending.config||{},patchFor(path));schedule();}

/* ---------- builders ---------- */
function addSlider(parent,o){
  const row=el('div','row');
  row.innerHTML=`<div class="rl"><span>${o.label}</span><span class="rv"><input type="number" step="${o.numStep||o.step}"><i>${o.unit||''}</i></span></div><input type="range" min="${o.min}" max="${o.max}" step="${o.step}">`;
  const num=$('input[type=number]',row),rng=$('input[type=range]',row);
  const paint=()=>{const p=(rng.value-rng.min)/(rng.max-rng.min)*100;rng.style.setProperty('--p',clamp(p,0,100)+'%');};
  const dec=o.dec==null?2:o.dec;
  const show=v=>{rng.value=v;num.value=(+v).toFixed(dec);paint();};
  const commit=v=>{setPath(CFG,o.path,v);if(o.custom)o.custom(v);else queuePatch(o.path);if(o.after)o.after();};
  rng.addEventListener('input',()=>{num.value=(+rng.value).toFixed(dec);paint();commit(+rng.value);});
  num.addEventListener('change',()=>{const v=parseFloat(num.value);if(isNaN(v))return;rng.value=v;paint();commit(v);});
  parent.appendChild(row);
  bindings.push({path:o.path,set:show});
  return {row,rng,num,show,paint};
}
function addToggle(parent,o){
  const row=el('label','toggle',`<span>${o.label}</span><input type="checkbox">`);
  const cb=$('input',row);
  cb.addEventListener('change',()=>{setPath(CFG,o.path,cb.checked);queuePatch(o.path);});
  parent.appendChild(row);bindings.push({path:o.path,set:v=>cb.checked=!!v});
}
function addSelect(parent,o){
  const row=el('div','toggle',`<span>${o.label}</span><select>${o.options.map(x=>`<option value="${x[0]}">${x[1]}</option>`).join('')}</select>`);
  const s=$('select',row);
  s.addEventListener('change',()=>{setPath(CFG,o.path,parseFloat(s.value));queuePatch(o.path);});
  parent.appendChild(row);bindings.push({path:o.path,set:v=>s.value=String(Math.round(v))});
}
function addNumber(parent,o){
  const row=el('div','toggle',`<span>${o.label}</span><span class="rv"><input type="number" step="${o.step||0.01}"><i>${o.unit||''}</i></span>`);
  const n=$('input',row);
  n.addEventListener('change',()=>{const v=parseFloat(n.value);if(isNaN(v))return;setPath(CFG,o.path,v);queuePatch(o.path);});
  parent.appendChild(row);bindings.push({path:o.path,set:v=>n.value=(+v).toFixed(o.dec==null?2:o.dec)});
}
function section(parent,title,open,note){
  const d=el('details','sec');if(open)d.open=true;
  d.innerHTML=`<summary>${title}</summary><div class="body">${note?`<div class="note">${note}</div>`:''}</div>`;
  parent.appendChild(d);return $('.body',d);
}
function button(parent,label,cls,fn){const b=el('button','btn '+(cls||''),label);b.addEventListener('click',fn);parent.appendChild(b);return b;}

/* ---------- pages ---------- */
const servoUI=[],poseUI={};
function buildManual(){
  const pg=$('[data-page=manual]');
  const s1=section(pg,'Servos — direct control',true,'Each slider moves one servo straight to that angle. The range is the hard limit set in Calibration.');
  s1.classList.add('manual-only');
  for(let i=0;i<3;i++){
    const row=el('div','row',`<div class="rl"><span>Servo ${i+1}</span><span class="rv"><input type="number" step="0.5"><i>°</i></span></div><input type="range" min="0" max="180" step="0.5"><div class="cap"><span></span><span></span></div>`);
    const rng=$('input[type=range]',row),num=$('input[type=number]',row);
    const paint=()=>rng.style.setProperty('--p',clamp((rng.value-rng.min)/(rng.max-rng.min)*100,0,100)+'%');
    const send=v=>{pending.servo[i]=v;schedule();};
    rng.addEventListener('pointerdown',()=>dragging['s'+i]=true);
    rng.addEventListener('change',()=>{dragging['s'+i]=false;});
    rng.addEventListener('input',()=>{num.value=(+rng.value).toFixed(1);paint();send(+rng.value);});
    num.addEventListener('focus',()=>dragging['s'+i]=true);
    num.addEventListener('change',()=>{dragging['s'+i]=false;const v=parseFloat(num.value);if(isNaN(v))return;rng.value=v;paint();send(+rng.value);});
    num.addEventListener('blur',()=>dragging['s'+i]=false);
    s1.appendChild(row);servoUI.push({row,rng,num,paint,cap:row.querySelectorAll('.cap span')});
  }
  const s2=section(pg,'Platform pose — inverse kinematics',true,'Move the whole platform: height, tilt and tilt direction. Servo angles are computed for you.');
  s2.classList.add('manual-only');
  [['h','Height',0,1,0.05,2,'cm'],['theta','Tilt θ',0,10,0.1,1,'°'],['phi','Direction φ',0,360,1,0,'°']].forEach(([k,label,mn,mx,st,dec,u])=>{
    const row=el('div','row',`<div class="rl"><span>${label}</span><span class="rv"><input type="number" step="${st}"><i>${u}</i></span></div><input type="range" min="${mn}" max="${mx}" step="${st}">`);
    const rng=$('input[type=range]',row),num=$('input[type=number]',row);
    const paint=()=>rng.style.setProperty('--p',clamp((rng.value-rng.min)/(rng.max-rng.min)*100,0,100)+'%');
    const send=()=>{pending.pose={h:+poseUI.h.rng.value,theta:+poseUI.theta.rng.value,phi:+poseUI.phi.rng.value};schedule();};
    rng.addEventListener('pointerdown',()=>dragging['p'+k]=true);
    rng.addEventListener('change',()=>dragging['p'+k]=false);
    rng.addEventListener('input',()=>{num.value=(+rng.value).toFixed(dec);paint();send();});
    num.addEventListener('focus',()=>dragging['p'+k]=true);
    num.addEventListener('blur',()=>dragging['p'+k]=false);
    num.addEventListener('change',()=>{const v=parseFloat(num.value);if(isNaN(v))return;rng.value=v;paint();send();});
    s2.appendChild(row);poseUI[k]={row,rng,num,paint,dec};
  });
  const b=el('div','btns');button(b,'Level platform at work height','',()=>api('/api/autolevel',{}).then(handle));s2.appendChild(b);
}

function buildAuto(){
  const pg=$('[data-page=auto]');
  const live=section(pg,'Live',true);
  live.innerHTML=`<div class="kv" id="autoKV"></div>`;
  const kv=$('#autoKV');
  [['Ball position (px)','a_pos'],['Target (px)','a_tgt'],['Error x / y (cm)','a_err'],['Distance (cm)','a_dist'],['Velocity x / y (cm/s)','a_vel'],['Speed (cm/s)','a_spd'],
   ['PID error (norm.)','a_perr'],['P term','a_p'],['I term','a_i'],['D term','a_d'],['Output tilt θ / φ','a_out'],['Work height (cm)','a_h']].forEach(([l,id])=>{kv.appendChild(el('div','',l));kv.appendChild(el('div','',`<span id="${id}">–</span>`));});
  const pid=section(pg,'PID',true,'Changes apply instantly. Errors are normalised to a 200-px-wide frame, so gains from the original project are a good starting point.');
  addSlider(pid,{path:'pid.kp',label:'Kp',min:0,max:0.03,step:0.0001,dec:5});
  addSlider(pid,{path:'pid.ki',label:'Ki',min:0,max:0.01,step:0.00005,dec:5});
  addSlider(pid,{path:'pid.kd',label:'Kd',min:0,max:0.03,step:0.0001,dec:5});
  addSlider(pid,{path:'pid.alpha',label:'α  output filter (1 = none)',min:0.05,max:1,step:0.01,dec:2});
  addSlider(pid,{path:'pid.beta',label:'β  tilt gain (tanh)',min:0.02,max:2,step:0.01,dec:2});
  const b=el('div','btns');button(b,'Reset PID state','',()=>api('/api/pid_reset',{}).then(()=>toast('PID state cleared')));pid.appendChild(b);
  const pt=section(pg,'Target path',true,'Moves the target around a shape centred on your target point while the PID keeps the ball on it. Size = distance from the centre to the farthest point. Start it while in Auto mode. For accurate sizes, run “Measure scale from ball” in Calibration.');
  const SH=[['circle','Circle','<circle cx="12" cy="12" r="8"/>'],['square','Square','<rect x="5" y="5" width="14" height="14"/>'],
    ['triangle','Triangle','<path d="M12 4 L20 19 H4 Z"/>'],
    ['star','Star','<path d="M12 3 L14.4 9.3 L21 9.6 L15.8 13.7 L17.6 20 L12 16.4 L6.4 20 L8.2 13.7 L3 9.6 L9.6 9.3 Z"/>'],
    ['line','Line','<path d="M4 18 L20 6"/>'],
    ['spiral','Spiral','<path d="M12 12 a1.5 1.5 0 0 1 3 0 a3.5 3.5 0 0 1 -7 0 a5.5 5.5 0 0 1 11 0 a7.5 7.5 0 0 1 -15 0"/>']];
  const grid=el('div','shapes');
  const markShape=()=>grid.querySelectorAll('button').forEach(b=>b.classList.toggle('on',b.dataset.shape===PAT.shape));
  SH.forEach(([k,lab,svg])=>{const b=el('button','',`<svg viewBox="0 0 24 24">${svg}</svg><span>${lab}</span>`);b.dataset.shape=k;b.addEventListener('click',()=>{PAT.shape=k;markShape();pushPattern();});grid.appendChild(b);});
  pt.appendChild(grid);
  const sSize=objSlider(pt,PAT,'size','Size',0.5,6,0.1,'cm',1,pushPattern);
  const sSpeed=objSlider(pt,PAT,'speed','Speed',0.5,10,0.1,'cm/s',1,pushPattern);
  const sRot=objSlider(pt,PAT,'rot','Rotation',0,360,5,'°',0,pushPattern);
  const dr=el('div','row','<div class="rl"><span>Direction (as seen on the video)</span></div>');
  const dseg=el('div','seg','<button data-d="cw">↻ Clockwise</button><button data-d="ccw">↺ Counter-clockwise</button>');
  const markDir=()=>dseg.querySelectorAll('button').forEach(b=>b.classList.toggle('on',b.dataset.d===PAT.direction));
  dseg.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;PAT.direction=b.dataset.d;markDir();pushPattern();});
  dr.appendChild(dseg);pt.appendChild(dr);
  pt.appendChild(el('div','note','<span id="patSafe"></span>'));
  const pbtn=el('div','btns');
  button(pbtn,'▶ Start path','primary',async()=>handle(await api('/api/pattern',{action:'start',...PAT})));
  button(pbtn,'■ Stop','',async()=>handle(await api('/api/pattern',{action:'stop'})));
  pt.appendChild(pbtn);
  pt.appendChild(el('div','pstat','<span class="dot" id="patDot"></span><span id="patTxt">Idle</span>'));
  markShape();markDir();
  patUI={size:sSize,speed:sSpeed,rot:sRot,markShape,markDir,safe:$('#patSafe',pt),dot:$('#patDot',pt),txt:$('#patTxt',pt)};
  const adv=section(pg,'Limits & orientation',false);
  addSlider(adv,{path:'pid.tilt_limit',label:'Max tilt the PID may use',min:1,max:30,step:0.5,dec:1,unit:'°'});
  adv.appendChild(el('div','note','<span id="tiltReal">–</span>'));
  addSlider(adv,{path:'pid.i_limit',label:'Integral clamp (anti-windup)',min:0,max:1,step:0.01,dec:2});
  addSlider(adv,{path:'pid.lost_timeout',label:'Level platform if ball lost for',min:0.1,max:2,step:0.05,dec:2,unit:'s'});
  addSelect(adv,{path:'pid.rot_deg',label:'Camera → platform rotation',options:[[0,'0°'],[90,'90°'],[180,'180°'],[270,'270°']]});
  addToggle(adv,{path:'pid.mirror',label:'Mirror camera Y axis'});
  adv.appendChild(el('div','note','If the ball rolls away from the target instead of towards it, change the rotation (try 90° ↔ 270°, or 0° ↔ 180°) or toggle mirror.'));
}

function updateHue(){
  const t=CFG&&CFG.tracking,bar=$('#hueBar');if(!t||!bar)return;
  const stops=[];for(let i=0;i<=12;i++)stops.push(`hsl(${i*30},75%,48%)`);
  const spectrum=`linear-gradient(to right,${stops.join(',')})`;
  const pa=t.h_min/180*100,pb=(t.h_max+1)/180*100,dk='rgba(9,13,19,.82)';
  const dim=t.h_min<=t.h_max
    ?`linear-gradient(to right,${dk} 0,${dk} ${pa}%,transparent ${pa}%,transparent ${pb}%,${dk} ${pb}%,${dk} 100%)`
    :`linear-gradient(to right,transparent 0,transparent ${pb}%,${dk} ${pb}%,${dk} ${pa}%,transparent ${pa}%,transparent 100%)`;
  bar.style.background=`${dim},${spectrum}`;
}
function applyTracking(obj){
  Object.assign(CFG.tracking,obj);
  pending.config=deepMerge(pending.config||{},{tracking:{...obj}});schedule();refreshBindings();
}
function localSlider(parent,label,min,max,step,key,unit){
  const row=el('div','row',`<div class="rl"><span>${label}</span><span class="rv"><b></b><i>${unit}</i></span></div><input type="range" min="${min}" max="${max}" step="${step}">`);
  const rng=$('input',row),b=$('b',row);
  const paint=()=>{rng.style.setProperty('--p',clamp((rng.value-min)/(max-min)*100,0,100)+'%');b.textContent=rng.value;};
  rng.value=PICK[key];paint();
  rng.addEventListener('input',()=>{PICK[key]=+rng.value;paint();});
  parent.appendChild(row);
}
function objSlider(parent,obj,key,label,min,max,step,unit,dec,onch){
  const row=el('div','row',`<div class="rl"><span>${label}</span><span class="rv"><input type="number" step="${step}"><i>${unit}</i></span></div><input type="range" min="${min}" max="${max}" step="${step}">`);
  const num=$('input[type=number]',row),rng=$('input[type=range]',row);
  const paint=()=>rng.style.setProperty('--p',clamp((rng.value-rng.min)/(rng.max-rng.min)*100,0,100)+'%');
  const show=v=>{rng.value=v;num.value=(+v).toFixed(dec);paint();};
  rng.addEventListener('input',()=>{obj[key]=+rng.value;num.value=(+rng.value).toFixed(dec);paint();if(onch)onch();});
  num.addEventListener('change',()=>{const v=parseFloat(num.value);if(isNaN(v))return;obj[key]=v;show(v);if(onch)onch();});
  parent.appendChild(row);show(obj[key]);
  return {rng,num,show,paint,row};
}
function pushPattern(){pending.pattern={...PAT};schedule();}
function syncPatUI(){if(!patUI)return;patUI.size.show(PAT.size);patUI.speed.show(PAT.speed);patUI.rot.show(PAT.rot);patUI.markShape();patUI.markDir();}

/* ---------- video size ---------- */
function setVZ(z){
  VZ=z;document.querySelectorAll('#sizeSeg button').forEach(b=>b.classList.toggle('on',b.dataset.z===z));
  $('#vwrap').style.setProperty('--vs',({S:0.6,M:0.8,L:1})[z]||1);
  try{localStorage.setItem('bb_vz',z);}catch(e){}
}

/* ---------- dialogs ---------- */
function modal(o){
  return new Promise(res=>{
    const bg=$('#modalBg'),m=$('.modal',bg);
    m.innerHTML=`<h3></h3><p></p>${o.input?'<input type="text" maxlength="40" spellcheck="false" autocomplete="off">':''}<div class="mb"><button class="btn sm" data-x="0">Cancel</button><button class="btn sm ${o.danger?'danger':'primary'}" data-x="1"></button></div>`;
    $('h3',m).textContent=o.title;$('p',m).textContent=o.text||'';$('[data-x="1"]',m).textContent=o.ok||'OK';
    if(!o.text)$('p',m).style.display='none';
    const inp=$('input',m);if(inp)inp.value=o.value||'';
    const val=()=>inp?(inp.value.trim()||null):true;
    const done=v=>{bg.classList.remove('on');bg.onclick=null;m.onclick=null;document.removeEventListener('keydown',key);res(v);};
    const key=e=>{if(e.key==='Escape')done(null);else if(e.key==='Enter')done(val());};
    m.onclick=e=>{const b=e.target.closest('button[data-x]');if(b)done(b.dataset.x==='1'?val():null);};
    bg.onclick=e=>{if(e.target===bg)done(null);};
    document.addEventListener('keydown',key);bg.classList.add('on');
    if(inp){inp.focus();inp.select();}
  });
}

/* ---------- tuning presets (PID + ball colour/size) ---------- */
function setActive(n){ACTIVE=n||null;try{if(n)localStorage.setItem('bb_active_preset',n);else localStorage.removeItem('bb_active_preset');}catch(e){}}
function snapCfg(){const o={};for(const sec in TUNKEYS){o[sec]={};TUNKEYS[sec].forEach(k=>{o[sec][k]=CFG[sec][k];});}return o;}
function sameVals(a,b){
  for(const sec in TUNKEYS)for(const k of TUNKEYS[sec]){
    const x=a[sec]&&a[sec][k],y=b[sec]&&b[sec][k];
    if(x==null||y==null)return false;
    if(Math.abs(x-y)>1e-9*Math.max(1,Math.abs(x),Math.abs(y)))return false;
  }
  return true;
}
const curPreset=()=>TUN.find(p=>p.name===ACTIVE)||null;
function renderPresets(){
  const sel=$('#tunSel'),sig=TUN.map(p=>p.name).join('\u0001')+'|'+(ACTIVE||'');
  if(sel.dataset.sig!==sig){
    sel.innerHTML='';
    const o0=document.createElement('option');o0.value='';o0.textContent=TUN.length?'— none loaded —':'— no presets saved yet —';sel.appendChild(o0);
    TUN.forEach(p=>{const o=document.createElement('option');o.value=p.name;o.textContent=p.name;sel.appendChild(o);});
    sel.value=curPreset()?ACTIVE:'';sel.dataset.sig=sig;
  }
  const cur=curPreset(),badge=$('#pBadge');
  if(!cur){badge.className='badge';badge.textContent=TUN.length?'none loaded':'empty';}
  else if(CFG&&TUNKEYS.pid&&!sameVals(snapCfg(),cur.values)){badge.className='badge mod';badge.textContent='● unsaved changes';}
  else{badge.className='badge ok';badge.textContent='✓ matches saved';}
  $('#pSave').disabled=!cur;$('#pRename').disabled=!cur;$('#pDelete').disabled=!cur;
}
async function refreshTun(){
  const r=await api('/api/presets');
  if(r&&r.presets){TUN=r.presets;TUNKEYS=r.keys;if(ACTIVE&&!TUN.some(p=>p.name===ACTIVE))setActive(null);renderPresets();}
}
async function flushNow(){flush();await new Promise(r=>setTimeout(r,160));}
async function loadPreset(name){
  const cur=curPreset();
  if(cur&&CFG&&!sameVals(snapCfg(),cur.values)){
    const ok=await modal({title:'Discard unsaved changes?',text:`“${cur.name}” has changes that are not saved. Loading another preset replaces the current values.`,ok:'Load anyway',danger:true});
    if(!ok){$('#tunSel').value=ACTIVE||'';return;}
  }
  const r=handle(await api('/api/presets/load',{name}));
  if(r&&r.config){CFG=r.config;DER=r.derived;setActive(r.name);refreshBindings();updateRanges();toast(`Loaded “${r.name}”`);}
  else $('#tunSel').value=ACTIVE||'';
  renderPresets();
}
async function savePreset(){
  const cur=curPreset();if(!cur){savePresetAs();return;}
  await flushNow();
  const r=handle(await api('/api/presets/save',{name:cur.name,overwrite:true}));
  if(r.ok){TUN=r.presets;toast(`Saved “${cur.name}”`);renderPresets();}
}
async function savePresetAs(){
  const name=await modal({title:'Save tuning preset',text:'Stores the PID values plus the ball colour and size settings under this name.',input:true,value:'',ok:'Save'});
  if(!name)return;
  await flushNow();
  let r=await api('/api/presets/save',{name,overwrite:false});
  if(r&&r.exists){
    const ok=await modal({title:'Overwrite preset?',text:`“${name}” already exists. Replace it with the current values?`,ok:'Overwrite',danger:true});
    if(!ok)return;
    r=await api('/api/presets/save',{name,overwrite:true});
  }
  if(handle(r).ok){TUN=r.presets;setActive(r.name);toast(`Saved “${r.name}”`);renderPresets();}
}
async function renamePreset(){
  const cur=curPreset();if(!cur)return;
  const nn=await modal({title:'Rename preset',input:true,value:cur.name,ok:'Rename'});
  if(!nn||nn===cur.name)return;
  const r=handle(await api('/api/presets/rename',{name:cur.name,new_name:nn}));
  if(r.ok){TUN=r.presets;setActive(r.name);toast(`Renamed to “${r.name}”`);renderPresets();}
}
async function deletePreset(){
  const cur=curPreset();if(!cur)return;
  const ok=await modal({title:'Delete preset?',text:`“${cur.name}” will be removed permanently. Your current settings are not changed.`,ok:'Delete',danger:true});
  if(!ok)return;
  const r=handle(await api('/api/presets/delete',{name:cur.name}));
  if(r.ok){TUN=r.presets;setActive(null);toast('Preset deleted');renderPresets();}
}

async function doPick(x,y){
  const r=await api('/api/pick_color',{x,y,hue_margin:PICK.hue,sv_margin:PICK.sv/100,set_size:PICK.size});
  if(!r||r.ok===false){toast((r&&r.error)||'Colour pick failed',true);return;}
  CFG.tracking=r.tracking;refreshBindings();
  const t=r.tracking;
  toast(`Colour learned: H ${t.h_min}–${t.h_max}${t.h_min>t.h_max?' (wraps)':''}, S ${t.s_min}–${t.s_max}, V ${t.v_min}–${t.v_max}. `+(r.found?'Ball tracked ✓':'Ball not found inside the detection area — check the Mask view.'),!r.found);
  setTool(null);
}
let hWork=null;const limUI=[];
function buildCal(){
  const pg=$('[data-page=calibration]');
  const o=section(pg,'1 · Servo offsets',true,'Fine-trim each servo so the platform is physically level with the platform at work height. Use a spirit level. Moving a slider re-levels the platform.');
  for(let i=0;i<3;i++)addSlider(o,{path:`servo.offsets.${i}`,label:`Servo ${i+1} offset`,min:-30,max:30,step:0.25,dec:2,unit:'°'});
  const ob=el('div','btns');button(ob,'Level platform','',()=>api('/api/autolevel',{}).then(handle).then(()=>{}));o.appendChild(ob);

  const l=section(pg,'2 · Hard servo limits',false,'The servo is moved to the value while you drag, so you can find where the linkage binds. Back off a little from the binding point. “Down” = platform lowest, “Up” = platform highest. The software will never command outside these.');
  for(let i=0;i<3;i++){
    ['down','up'].forEach(w=>{
      const key=w==='down'?'servo.limit_down':'servo.limit_up';
      const s=addSlider(l,{path:`${key}.${i}`,label:`Servo ${i+1} — ${w==='down'?'Down limit (min)':'Up limit (max)'}`,min:0,max:180,step:1,dec:0,unit:'°',
        custom:v=>{pending.limit[`${i}${w}`]={index:i,which:w,value:v};schedule();updateServoRanges();}});
    });
  }

  const hh=section(pg,'3 · Work height',true,'Height of the platform centre while balancing (Auto mode always uses this height).');
  hWork=addSlider(hh,{path:'platform.h_work',label:'Work height',min:8,max:13,step:0.05,dec:2,unit:'cm'});

  const g=section(pg,'4 · Center, target & detection area',true,'Pick a tool, then click or drag on the video. Detection area = only ball candidates inside this circle are accepted.');
  const tb=el('div','btns');
  [['center','Set center'],['target','Set target'],['radius','Detection radius']].forEach(([k,lab])=>{const b=button(tb,lab,'',()=>setTool(k));b.dataset.tool=k;});
  g.appendChild(tb);
  const kv=el('div','kv','');kv.innerHTML='<div>Center (px)</div><div id="c_center">–</div><div>Target (px)</div><div id="c_target">–</div>';kv.style.margin='12px 0';g.appendChild(kv);
  addSlider(g,{path:'camera.detection_radius',label:'Detection radius',min:10,max:200,step:1,dec:0,unit:'px'});
  const gb=el('div','btns');
  button(gb,'Target = center','',()=>{CFG.camera.target=CFG.camera.center.slice();queuePatch('camera.target');updateCam();});
  button(gb,'Measure scale from ball','',async()=>{const r=handle(await api('/api/scale_from_ball',{}));if(r.ok){CFG=r.config;refreshBindings();toast(`Scale: ${fmt(r.px_per_cm,2)} px/cm (ball r ≈ ${fmt(r.radius_px,1)} px)`);}});
  g.appendChild(gb);
  addNumber(g,{path:'camera.px_per_cm',label:'Scale',step:0.1,dec:2,unit:'px/cm'});
  addNumber(g,{path:'camera.ball_diameter_cm',label:'Ball diameter',step:0.1,dec:2,unit:'cm'});

  const t=section(pg,'5 · Ball colour & tracking',true,'Easiest: press “Pick ball colour”, then click the middle of the ball on the video. Then check the “Mask” view — the ball should be one clean bright blob and nothing else.');
  const pk=el('div','btns');
  const pb=button(pk,'🎯 Pick ball colour','primary',()=>setTool('pick'));pb.dataset.tool='pick';
  button(pk,'Show mask','',()=>setView('mask'));
  t.appendChild(pk);
  const ps=el('div','toggle','<span>Quick colour</span><select><option value="">Choose a colour…</option>'+Object.keys(PRESETS).map(k=>`<option>${k}</option>`).join('')+'</select>');
  $('select',ps).addEventListener('change',e=>{const p=PRESETS[e.target.value];if(p){applyTracking(p);toast('Preset applied — fine-tune with the sliders or pick from the video');}e.target.value='';});
  t.appendChild(ps);
  localSlider(t,'Picker: hue margin',2,20,1,'hue','±');
  localSlider(t,'Picker: saturation/brightness margin',5,50,1,'sv','%');
  const sz=el('label','toggle','<span>Picker also sets ball size limits</span><input type="checkbox" checked>');
  $('input',sz).addEventListener('change',e=>PICK.size=e.target.checked);t.appendChild(sz);
  t.appendChild(el('div','','<div id="hueBar" title="Bright = hues that are accepted"></div>'));
  t.appendChild(el('div','note','Hue bar: bright part = accepted hues. Hue min above Hue max means the range wraps around through red.'));
  const A=updateHue;
  addSlider(t,{path:'tracking.h_min',label:'Hue start',min:0,max:179,step:1,dec:0,after:A});
  addSlider(t,{path:'tracking.h_max',label:'Hue end',min:0,max:179,step:1,dec:0,after:A});
  addSlider(t,{path:'tracking.s_min',label:'Saturation min',min:0,max:255,step:1,dec:0});
  addSlider(t,{path:'tracking.s_max',label:'Saturation max',min:0,max:255,step:1,dec:0});
  addSlider(t,{path:'tracking.v_min',label:'Value (brightness) min',min:0,max:255,step:1,dec:0});
  addSlider(t,{path:'tracking.v_max',label:'Value (brightness) max',min:0,max:255,step:1,dec:0});
  addSlider(t,{path:'tracking.blur',label:'Blur kernel',min:1,max:11,step:2,dec:0});
  addSlider(t,{path:'tracking.erode',label:'Erode (remove specks)',min:0,max:5,step:1,dec:0});
  addSlider(t,{path:'tracking.dilate',label:'Dilate (fill gaps)',min:0,max:5,step:1,dec:0});
  addSlider(t,{path:'tracking.min_radius',label:'Min ball radius',min:1,max:100,step:0.5,dec:1,unit:'px'});
  addSlider(t,{path:'tracking.max_radius',label:'Max ball radius',min:5,max:150,step:0.5,dec:1,unit:'px'});
  addSlider(t,{path:'tracking.min_circularity',label:'Min circularity',min:0,max:1,step:0.01,dec:2});
  const tb2=el('div','btns');
  button(tb2,'Reset size & filter defaults','',()=>applyTracking({blur:3,erode:1,dilate:1,min_radius:6,max_radius:45,min_circularity:0.6}));
  t.appendChild(tb2);

  const c=section(pg,'6 · Camera exposure',false,'Fast-moving ball: a short manual exposure removes motion blur (needs good lighting / higher gain).');
  addToggle(c,{path:'camera.awb_auto',label:'Auto white balance (off = colours stay stable)'});
  addToggle(c,{path:'camera.exposure_auto',label:'Auto exposure'});
  addSlider(c,{path:'camera.exposure_us',label:'Exposure time',min:200,max:16000,step:100,dec:0,unit:'µs'});
  addSlider(c,{path:'camera.gain',label:'Analogue gain',min:1,max:16,step:0.1,dec:1});

  const r=section(pg,'7 · Robot geometry (advanced)',false,'Arm lengths in cm. Wrong values give wrong kinematics.');
  addNumber(r,{path:'geometry.LP',label:'LP  platform radius',unit:'cm'});
  addNumber(r,{path:'geometry.L1',label:'L1  upper arm',unit:'cm'});
  addNumber(r,{path:'geometry.L2',label:'L2  lower arm (horn)',unit:'cm'});
  addNumber(r,{path:'geometry.LB',label:'LB  base radius',unit:'cm'});
  addToggle(r,{path:'geometry.invert',label:'Invert arms (elbows inward)'});

  const sb=el('div','btns');button(sb,'Save all settings','primary',saveAll);pg.appendChild(sb);
}

/* ---------- state ↔ UI ---------- */
function refreshBindings(){bindings.forEach(b=>b.set(getPath(CFG,b.path)));updateHue();updateCam();updateServoRanges();}
function updateRanges(){
  if(!DER||!DER.h_range)return;
  const [lo,hi]=DER.h_range;
  if(poseUI.h){poseUI.h.rng.min=lo;poseUI.h.rng.max=hi;}
  if(hWork){hWork.rng.min=lo;hWork.rng.max=hi;hWork.paint();}
  if(poseUI.theta){poseUI.theta.rng.max=Math.max(1,DER.theta_max||10);}
}
function updateServoRanges(){
  if(!CFG)return;
  servoUI.forEach((u,i)=>{
    const d=CFG.servo.limit_down[i],up=CFG.servo.limit_up[i];
    u.rng.min=Math.min(d,up);u.rng.max=Math.max(d,up);
    const lowIsUp=up<d;
    u.cap[0].textContent=`${u.rng.min}° ${lowIsUp?'(up)':'(down)'}`;u.cap[1].textContent=`${u.rng.max}° ${lowIsUp?'(down)':'(up)'}`;
    u.paint();
  });
}
function updateCam(){
  if(!CFG)return;
  const c=$('#c_center'),t=$('#c_target');
  if(c)c.textContent=CFG.camera.center.map(v=>fmt(v,1)).join(', ');
  if(t)t.textContent=CFG.camera.target.map(v=>fmt(v,1)).join(', ');
  bindings.forEach(b=>{if(b.path==='camera.detection_radius')b.set(CFG.camera.detection_radius);});
}
async function saveAll(){const r=handle(await api('/api/save',{}));if(r.ok)toast('Settings saved');}
function setConn(ok){connOK=ok;$('#dConn').className='dot'+(ok?' ok':'');$('#pConn').textContent=ok?'connected':'offline';}

const TILES=[['Ball position','t_pos'],['Error from target','t_err'],['Velocity','t_vel'],['Platform tilt','t_tilt'],['Control loop','t_loop'],['Servo angles','t_servo']];
TILES.forEach(([l,id])=>$('#tiles').appendChild(el('div','card tile',`<div class="l">${l}</div><div class="v" id="${id}">–</div><div class="s" id="${id}s">&nbsp;</div>`)));
const T=(id,v,s,cls)=>{const a=$('#'+id);a.textContent=v;a.className='v'+(cls?' '+cls:'');$('#'+id+'s').textContent=s||'\u00a0';};
const S=(id,v)=>{const e=$('#'+id);if(e)e.textContent=v;};

function applyMode(m){
  MODE=m;document.body.dataset.mode=m;
  document.querySelectorAll('#modeSeg button').forEach(b=>b.classList.toggle('on',b.dataset.m===m));
  document.querySelectorAll('.page').forEach(p=>p.classList.toggle('on',p.dataset.page===m));
  if(m!=='calibration')setTool(null);
}
function render(){
  const s=ST,b=s.ball,p=s.pose;
  $('#pMode').textContent=s.mode==='transition'?'moving…':s.mode;
  if(s.mode!=='transition'&&Date.now()>modeLock&&s.mode!==MODE)applyMode(s.mode);
  $('#pSim').style.display=s.sim?'':'none';$('#btnKick').style.display=s.sim?'':'none';
  $('#dBall').className='dot'+(b.found?' ok':(b.has_pos?' warn':''));
  $('#pBall').textContent=b.found?'ball tracked':'ball not found';

  T('t_pos',b.has_pos?`${fmt(b.x,1)}, ${fmt(b.y,1)}`:'–',b.found?`radius ${fmt(b.r,1)} px`:'not found',b.found?'':'bad');
  T('t_err',`${fmt(b.dist_cm,2)} cm`,`Δx ${fmt(b.ex_cm,2)}   Δy ${fmt(b.ey_cm,2)} cm`);
  T('t_vel',`${fmt(b.speed_cm,1)} cm/s`,`vx ${fmt(b.vx_cm,1)}   vy ${fmt(b.vy_cm,1)}`);
  T('t_tilt',`${fmt(p.theta,1)}°`,`φ ${fmt(p.phi,0)}°   h ${fmt(p.h,2)} cm`);
  T('t_loop',`${fmt(s.fps,0)} Hz`,`${fmt(s.proc_ms,1)} ms / frame`,s.fps<25?'amber':'');
  const cl=s.servo.clamped.map((c,i)=>c?`S${i+1}`:null).filter(Boolean);
  T('t_servo',s.servo.angles.map(a=>fmt(a,0)).join(' / '),cl.length?`limit reached: ${cl.join(', ')}`:'within limits',cl.length?'amber':'');

  S('a_pos',b.has_pos?`${fmt(b.x,1)}, ${fmt(b.y,1)}`:'–');
  S('a_err',`${fmt(b.ex_cm,2)} / ${fmt(b.ey_cm,2)}`);S('a_dist',fmt(b.dist_cm,2));
  S('a_vel',`${fmt(b.vx_cm,1)} / ${fmt(b.vy_cm,1)}`);S('a_spd',fmt(b.speed_cm,1));
  const q=s.pid||{};
  const v2=a=>a?`${fmt(a[0],3)} / ${fmt(a[1],3)}`:'–';
  S('a_perr',q.err?`${fmt(q.err[0],1)} / ${fmt(q.err[1],1)}`:'–');S('a_p',v2(q.p));S('a_i',v2(q.i));S('a_d',v2(q.d));
  S('a_out',q.theta!=null?`${fmt(q.theta,2)}° / ${fmt(q.phi,0)}°`:'–');S('a_h',fmt(p.h,2));
  S('a_tgt',s.target?`${fmt(s.target[0],1)}, ${fmt(s.target[1],1)}`:'–');
  S('tiltReal',`Real limit at the ${fmt(p.h_work,1)} cm work height: ${fmt(p.theta_max,1)}°  →  effective max tilt = ${fmt(Math.min(CFG.pid.tilt_limit,p.theta_max),1)}° (the smaller of the two)`);
  if(patUI&&s.pattern){
    const g=s.pattern;
    if(!PATSYNC){PATSYNC=true;if(g.active){Object.assign(PAT,{shape:g.shape,size:g.size,speed:g.speed,direction:g.direction,rot:g.rot});syncPatUI();}}
    const mx=Math.max(0.5,Math.min(6,g.safe_size));
    patUI.size.rng.max=mx;
    if(PAT.size>mx+1e-6){PAT.size=mx;patUI.size.show(mx);pushPattern();}
    patUI.safe.textContent=`Largest size that stays inside the detection area: ${fmt(g.safe_size,1)} cm`;
    patUI.txt.textContent=({idle:'Idle',approach:'Moving to the start of the path…',run:`Running — ${g.shape}`,return:'Returning to the target point…'})[g.phase]||'Idle';
    patUI.dot.className='dot'+(g.phase==='run'?' ok':(g.phase==='idle'?'':' warn'));
  }
  renderPresets();

  servoUI.forEach((u,i)=>{
    u.row.classList.toggle('clamped',!!s.servo.clamped[i]);
    if(!dragging['s'+i]){u.rng.value=s.servo.angles[i];u.num.value=fmt(s.servo.angles[i],1);u.paint();}
  });
  DER.h_range=[p.h_min,p.h_max];DER.theta_max=p.theta_max;updateRanges();
  [['h',p.h],['theta',p.theta],['phi',p.phi]].forEach(([k,v])=>{
    const u=poseUI[k];if(u&&!dragging['p'+k]){u.rng.value=v;u.num.value=fmt(v,u.dec);u.paint();}
  });
}
async function poll(){
  const s=await api('/api/state');
  if(!s||s.ok===false){setConn(false);return;}
  setConn(true);ST=s;if(CFG)render();
}

/* ---------- video tools ---------- */
function setTool(t){
  TOOL=(t&&TOOL===t)?null:t;
  document.querySelectorAll('[data-tool]').forEach(b=>b.classList.toggle('on',b.dataset.tool===TOOL));
  $('#vover').classList.toggle('active',!!TOOL);
  $('#toolhint').textContent=TOOL?{center:'Click or drag to place the CENTER',target:'Click or drag to place the TARGET',radius:'Click or drag to set the DETECTION RADIUS',pick:'Click the middle of the BALL to learn its colour'}[TOOL]:'';
}
const vover=$('#vover');let pressing=false;
function applyPointer(e){
  if(!TOOL||!CFG)return;
  const r=vover.getBoundingClientRect(),W=CFG.camera.work[0],H=CFG.camera.work[1];
  const x=+clamp((e.clientX-r.left)/r.width*W,0,W).toFixed(1),y=+clamp((e.clientY-r.top)/r.height*H,0,H).toFixed(1);
  const c=CFG.camera;
  if(TOOL==='center'){const same=c.target[0]===c.center[0]&&c.target[1]===c.center[1];c.center=[x,y];queuePatch('camera.center');if(same){c.target=[x,y];queuePatch('camera.target');}}
  else if(TOOL==='target'){c.target=[x,y];queuePatch('camera.target');}
  else{c.detection_radius=+Math.max(10,Math.hypot(x-c.center[0],y-c.center[1])).toFixed(1);queuePatch('camera.detection_radius');}
  updateCam();
}
vover.addEventListener('pointerdown',e=>{if(!TOOL)return;
  if(TOOL==='pick'){const r=vover.getBoundingClientRect();doPick((e.clientX-r.left)/r.width*CFG.camera.work[0],(e.clientY-r.top)/r.height*CFG.camera.work[1]);return;}
  pressing=true;vover.setPointerCapture(e.pointerId);applyPointer(e);});
vover.addEventListener('pointermove',e=>{if(pressing)applyPointer(e);});
vover.addEventListener('pointerup',()=>pressing=false);
vover.addEventListener('pointercancel',()=>pressing=false);

function setView(v){
  document.querySelectorAll('#viewSeg button').forEach(b=>b.classList.toggle('on',b.dataset.v===v));
  $('#video').src='/video_feed?view='+v+'&t='+Date.now();
}
$('#viewSeg').addEventListener('click',e=>{const b=e.target.closest('button');if(b)setView(b.dataset.v);});
$('#modeSeg').addEventListener('click',async e=>{
  const b=e.target.closest('button');if(!b)return;
  modeLock=Date.now()+2500;applyMode(b.dataset.m);
  handle(await api('/api/mode',{mode:b.dataset.m}));modeLock=Date.now()+300;
});
$('#btnLevel').addEventListener('click',async()=>{modeLock=Date.now()+1500;applyMode('manual');handle(await api('/api/autolevel',{}));modeLock=Date.now()+300;toast('Platform levelled');});
$('#btnSave').addEventListener('click',saveAll);
$('#btnKick').addEventListener('click',()=>api('/api/sim/kick',{}));
$('#sizeSeg').addEventListener('click',e=>{const b=e.target.closest('button');if(b)setVZ(b.dataset.z);});
$('#tunSel').addEventListener('change',e=>{const v=e.target.value;if(v)loadPreset(v);else{setActive(null);renderPresets();}});
$('#pSave').addEventListener('click',savePreset);
$('#pSaveAs').addEventListener('click',savePresetAs);
$('#pRename').addEventListener('click',renamePreset);
$('#pDelete').addEventListener('click',deletePreset);

/* ---------- boot ---------- */
(async function(){
  buildManual();buildAuto();buildCal();
  const r=await api('/api/config');
  if(!r||!r.config){toast('Cannot reach the robot controller',true);setConn(false);}
  else{CFG=r.config;DER=r.derived;refreshBindings();updateRanges();await refreshTun();}
  setVZ(VZ);setView('overlay');
  poll();setInterval(poll,130);
})();
})();
</script></body></html>
"""

if __name__ == "__main__":
    main()
