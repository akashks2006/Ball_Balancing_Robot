"""
sim_server.py — the final closed-loop simulation.

Owns the entire ball physics (rolling on a tilted plate,
5/7 rolling-without-slipping factor), the PID controller, and the REAL
inverse-kinematics class from robotKinematics.py

Runs a continuous physics loop and broadcasts state for rendering (30Hz). 

The browser is now a pure renderer + control panel:
It sends back force-impulse requests, resets, and live PID gain changes.

Loop each physics step:
  ball (x, y, vx, vy)
    --> PID.pid(target, current, velocity) --> (theta, phi)
    --> RobotKinematics.solve_inverse_kinematics_spherical(theta, phi, h)
    --> real servo angles + real 3D joint positions (A/B/C nodes)
    --> ball physics stepped using the resulting plate tilt
    --> broadcast full state to browser for rendering

"""

import asyncio
import json
import math
import os
import random
import time
from datetime import datetime

import websockets

import matplotlib
matplotlib.use("Agg")  # headless-safe backend; we only ever save PNGs, never show()
import matplotlib.pyplot as plt

from PID import PIDcontroller
from robotKinematics import RobotKinematics

# Where recorded PID-response graphs get saved. Created automatically if it
# doesn't exist yet.
GRAPH_OUTPUT_DIR = r"D:\Arduino Projects\Ball Balancing Robot\Python Simulation\v7 (Added Graph Plots)\Graphs"

# PHYSICAL CONSTANTS & SIMULATION PARAMETERS

G = 9810.0                         # mm/s^2  (9.81 m/s^2), gravitational acceleration
ROLL_FACTOR = 5 / 7                # Rolling ball approximation for no-slip motion on a tilted plate
DAMPING_PER_SEC = 0.95             # Per-second damping factor used to reduce velocity over time
RESTITUTION = 0.35                 # Bounce reduction when the ball impacts the platform rim
BALL_RADIUS = 7.0                  # mm, effective ball radius used for boundary checks
FORCE_MIN, FORCE_MAX = 200.0, 350.0  # mm/s — random disturbance impulse range for testing

DT = 1 / 120                       # Physics substep duration; this keeps integration stable
BROADCAST_HZ = 30                  # Browser rendering refresh rate in frames/sec
PRINT_HZ = 5                       # Console heartbeat rate for status text


class SimState:
    def __init__(self):
        # platform radius=60, base radius=75
        # actuated arm=50, passive arm=80

        self.model = RobotKinematics(lp=60.0, l1=80.0, l2=50.0, lb=75.0)
        self.pid = PIDcontroller(kp=0.0, ki=0.0, kd=0.0,
                                  alpha=0.2, beta=0.3,
                                  max_theta=self.model.maxtheta, conversion="linear")
        self.target = (0.0, 0.0)
        self.last_servo_deg = [0.0, 0.0, 0.0]
        self.last_pid = (0.0, 0.0)
        self.last_requested_theta = 0.0

        # --- recording state, for the "save PID graph" feature ---
        # record_log holds (t, x, y) samples in simulation time, not wall
        # clock, so the plotted x-axis is unaffected by any hiccups in the
        # asyncio loop's real-world timing.
        self.recording = False
        self.record_log = []
        self.record_elapsed = 0.0
        self.record_start_gains = (0.0, 0.0, 0.0)

        self.reset_ball()

    def reset_ball(self):
        self.x = self.y = 0.0
        self.vx = self.vy = 0.0
        # Clear PID internal state too, so old integral windup / filter
        # history doesn't carry over into the fresh run.
        self.pid.sum_err_x = self.pid.sum_err_y = 0.0
        self.pid.prev_out_x = self.pid.prev_out_y = 0.0
        self.pid.last_time = None

    def start_recording(self):
        # Fresh log every time recording starts; gains are snapshotted here
        # (rather than at save time) so the label on the graph reflects what
        # was actually driving the ball during the recorded run, even if the
        # sliders get nudged afterward.
        self.recording = True
        self.record_log = []
        self.record_elapsed = 0.0
        self.record_start_gains = (self.pid.kp, self.pid.ki, self.pid.kd)
        print("[record] started")

    def stop_recording(self):
        self.recording = False
        print(f"[record] stopped ({len(self.record_log)} samples, "
              f"{self.record_elapsed:.2f}s)")

    def save_graph(self):
        """Render the last recorded run as a PID-response graph (position vs
        time) and save it as a PNG to GRAPH_OUTPUT_DIR. Returns the full path
        of the saved file. Raises ValueError if there's nothing to plot.
        """
        if len(self.record_log) < 2:
            raise ValueError("No recorded data — press Start then Stop "
                              "Recording while the ball moves, then Save Graph.")

        ts = [p[0] for p in self.record_log]
        xs = [p[1] for p in self.record_log]
        ys = [p[2] for p in self.record_log]
        kp, ki, kd = self.record_start_gains
        tx, ty = self.target

        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        ax.plot(ts, xs, color="#d1495b", linewidth=1.7, label="X position")
        ax.plot(ts, ys, color="#2f6690", linewidth=1.7, label="Y position")
        ax.axhline(tx, color="#d1495b", linestyle="--", linewidth=1.1,
                   alpha=0.6, label="X target")
        ax.axhline(ty, color="#2f6690", linestyle="--", linewidth=1.1,
                   alpha=0.6, label="Y target")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position (mm)")
        ax.set_title("Ball Position vs. Time — PID Response")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", framealpha=0.9)
        ax.margins(x=0.01)

        gain_text = f"Kp = {kp:.3f}\nKi = {ki:.3f}\nKd = {kd:.3f}"
        ax.text(0.02, 0.98, gain_text, transform=ax.transAxes, fontsize=10,
                va="top", ha="left", family="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9,
                           edgecolor="#999999"))

        fig.tight_layout()

        os.makedirs(GRAPH_OUTPUT_DIR, exist_ok=True)
        filename = f"pid_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(GRAPH_OUTPUT_DIR, filename)
        fig.savefig(filepath)
        plt.close(fig)
        return filepath

    def apply_random_force(self):
        angle = random.uniform(0, 2 * math.pi)
        mag = random.uniform(FORCE_MIN, FORCE_MAX)
        self.vx += mag * math.cos(angle)
        self.vy += mag * math.sin(angle)
        print(f"[force] impulse {mag:.2f} in/s @ {math.degrees(angle):5.0f} deg")

    def step(self, dt):
        # --- control: PID -> desired (theta, phi) -> REAL inverse kinematics ---
        theta_cmd, phi = self.pid.pid(self.target, (self.x, self.y), (self.vx, self.vy))
        self.last_requested_theta = theta_cmd
        self.model.solve_inverse_kinematics_spherical(theta_cmd, phi, self.model.h)
        self.last_pid = (theta_cmd, phi)

        # Actual servo command angles — identical conversion to controller.py's
        # Goto_time_spherical, i.e. what would really be sent to the servos.
        self.last_servo_deg = [
            math.degrees(math.pi * 0.5 - self.model.theta1),
            math.degrees(math.pi * 0.5 - self.model.theta2),
            math.degrees(math.pi * 0.5 - self.model.theta3),
        ]

        # --- plant: ball rolling on the resulting plate orientation ---
        # theta/phi define the plate's unit normal vector (a, b, c) —
        # same convention robotKinematics.py itself uses internally.
        theta_r = math.radians(theta_cmd)
        phi_r = math.radians(phi)
        a = math.sin(theta_r) * math.cos(phi_r)
        b = math.sin(theta_r) * math.sin(phi_r)
        c = math.cos(theta_r)

        # In-plane component of gravity: g - (g.n)n, keeping only x,y
        # (standard small-tilt planar approximation for ball-on-plate,
        # matching how your real camera only ever sees a flat x,y position).
        gz = -G
        dot = gz * c
        g_in_x = 0.0 - dot * a
        g_in_y = 0.0 - dot * b

        ax = ROLL_FACTOR * g_in_x
        ay = ROLL_FACTOR * g_in_y

        damp = DAMPING_PER_SEC ** dt   # frame-rate independent damping
        self.vx = (self.vx + ax * dt) * damp
        self.vy = (self.vy + ay * dt) * damp
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Rim collision against the real platform radius
        max_r = self.model.lp - BALL_RADIUS
        r = math.hypot(self.x, self.y)
        if r > max_r and r > 0:
            nx, ny = self.x / r, self.y / r
            self.x, self.y = nx * max_r, ny * max_r
            v_dot_n = self.vx * nx + self.vy * ny
            if v_dot_n > 0:
                self.vx -= (1 + RESTITUTION) * v_dot_n * nx
                self.vy -= (1 + RESTITUTION) * v_dot_n * ny

        # --- recording tap: append a (t, x, y) sample if currently recording ---
        if self.recording:
            self.record_elapsed += dt
            self.record_log.append((self.record_elapsed, self.x, self.y))

    def state_payload(self):
        m = self.model
        return {
            "type": "state",
            "ball": {"x": self.x, "y": self.y, "vx": self.vx, "vy": self.vy},
            "pid": {"theta": self.last_pid[0], "phi": self.last_pid[1],
                     "requested_theta": self.last_requested_theta,
                     "max_theta": m.maxtheta,
                     "kp": self.pid.kp, "ki": self.pid.ki, "kd": self.pid.kd},
            "servo_deg": self.last_servo_deg,
            "points": {
                "A1": m.A1, "A2": m.A2, "A3": m.A3,
                "B1": m.B1, "B2": m.B2, "B3": m.B3,
                "C1": m.C1, "C2": m.C2, "C3": m.C3,
            },
            "dims": {"lp": m.lp, "lb": m.lb, "h": m.h},
            "recording": {
                "active": self.recording,
                "elapsed": self.record_elapsed,
                "samples": len(self.record_log),
            },
        }


# Shared simulation state used by the physics loop and all browser clients.
state = SimState()
clients = set()  # Active websocket clients that receive state broadcasts.
_last_print = 0.0


async def physics_loop():
    # Continuous simulation loop. This is the heart of the control system:
    # compute the plate tilt, update the ball's motion, and keep the state in sync.
    global _last_print
    while True:
        state.step(DT)
        now = time.perf_counter()

        # Print a concise status line at a lower rate than the physics loop to avoid
        # flooding the console while still providing real-time feedback.
        if now - _last_print >= 1.0 / PRINT_HZ:
            _last_print = now
            th, ph = state.last_pid
            s = state.last_servo_deg
            print(
                f"ball=({state.x:6.2f},{state.y:6.2f}) in  "
                f"vel=({state.vx:6.2f},{state.vy:6.2f}) in/s  |  "
                f"PID: theta={th:5.2f} phi={ph:6.1f}  |  "
                f"servo(deg)=[{s[0]:6.1f}, {s[1]:6.1f}, {s[2]:6.1f}]"
            )
        await asyncio.sleep(DT)


async def broadcast_loop():
    # Send the current simulation state to every connected browser at a fixed rate.
    # The frontend uses this data to render the moving ball and UI.
    while True:
        if clients:
            msg = json.dumps(state.state_payload())
            await asyncio.gather(*(c.send(msg) for c in list(clients)), return_exceptions=True)
        await asyncio.sleep(1 / BROADCAST_HZ)


async def handler(websocket):
    # Accept a browser websocket and register it as a live client for state updates.
    clients.add(websocket)
    print(f"Browser connected ({len(clients)} total)")
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                # Ignore malformed messages from the browser without crashing the server.
                continue

            mtype = data.get("type")
            if mtype == "force":
                # Browser requests a random disturbance impulse to test controller response.
                state.apply_random_force()
            elif mtype == "reset":
                # Reset the ball to the center and clear controller history.
                state.reset_ball()
                print("[reset] ball returned to origin")
            elif mtype == "gains":
                # Live tuning from the UI: update PID gains without restarting the server.
                state.pid.kp = float(data.get("kp", state.pid.kp))
                state.pid.ki = float(data.get("ki", state.pid.ki))
                state.pid.kd = float(data.get("kd", state.pid.kd))
                print(f"[gains] kp={state.pid.kp:.5f} ki={state.pid.ki:.6f} kd={state.pid.kd:.5f}")
            elif mtype == "record_start":
                # Begin logging (t, x, y) samples for the PID-response graph.
                state.start_recording()
                await websocket.send(json.dumps({"type": "record_status", "recording": True}))
            elif mtype == "record_stop":
                state.stop_recording()
                await websocket.send(json.dumps({
                    "type": "record_status",
                    "recording": False,
                    "samples": len(state.record_log),
                }))
            elif mtype == "save_graph":
                # Render the last recorded run to a PNG on disk.
                try:
                    path = state.save_graph()
                    print(f"[graph] saved to {path}")
                    await websocket.send(json.dumps({"type": "graph_saved", "ok": True, "path": path}))
                except Exception as e:
                    print(f"[graph] failed: {e}")
                    await websocket.send(json.dumps({"type": "graph_saved", "ok": False, "error": str(e)}))
    finally:
        clients.discard(websocket)
        print(f"Browser disconnected ({len(clients)} total)")


async def main():
    # Start the websocket server and launch the background physics and broadcast tasks.
    async with websockets.serve(handler, "localhost", 8765):
        m = state.model
        print("sim_server running: ws://localhost:8765")
        print(f"Robot dims: lp={m.lp} l1={m.l1} l2={m.l2} lb={m.lb}  "
              f"working height h={m.h:.2f}  max_theta={m.maxtheta:.2f} deg")
        print("Waiting for the browser to connect ...")
        await asyncio.gather(physics_loop(), broadcast_loop(), asyncio.Future())


if __name__ == "__main__":
    # Entry point for running the simulation as a standalone service.
    asyncio.run(main())
