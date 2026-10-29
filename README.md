# Ball-Balancing Robot — 3RRS Stewart Platform

A PID-controlled ball-balancing robot built on a 3-RRS parallel platform. A camera tracks the ball, a PID controller computes the tilt needed to hold it on target, and inverse kinematics turns that tilt into three servo angles.

This repo contains both stages of the project:

- **`Simulation/`**: a physics simulation with a live 3D viewer, used to develop and test the kinematics and control before the hardware existed.
- **`Hardware/`**: the final control code running on a Raspberry Pi, with a browser UI for live tuning and calibration.


![Demo](/Images/demo.gif)

---

## How it works

The **control layer** (PID) and the **kinematics layer** (IK) are kept deliberately separate:

- **PID** works only in terms of the ball. Its input is the position error, and its output is a desired platform tilt.
- **IK** works only in terms of geometry. Its input is a desired tilt, and its output is the servo angles that produce it.

Either layer can be swapped or tuned without touching the other.

### Inverse kinematics chain

1. Desired tilt → platform unit normal vector
2. Base joint positions
3. Top platform joint positions (plane–circle intersection)
4. Elbow positions (dual sphere-distance constraints)
5. Servo angles

### System Model

From the simulation, the ball-on-plate dynamics per axis reduce to:

$$G(s) = \frac{7007.1}{s\,(s + 0.0513)} \quad \text{(mm/rad)}$$

---

## Simulation (`Simulation/`)

A Python physics server paired with a Three.js browser viewer.

| File | Role |
|---|---|
| `sim_server.py` | Owns physics, PID and IK. Broadcasts state over WebSocket at 30 Hz |
| `platform_viewer.html` | Three.js 3D viewer that only renders state. No physics runs in the browser |
| `PID.py` | PID controller outputting tilt as spherical coordinates (θ, φ) |
| `robotKinematics.py` | `RobotKinematics` class that solves the IK |

**Features**
- Live 3D view of the platform, linkages and ball
- PID gain sliders (Kp, Ki, Kd)
- Recording of X(t) / Y(t) against the target, with export to timestamped PNG graphs

**Run**
```bash
cd Simulation
pip install -r requirements.txt
python sim_server.py
```
Then open `platform_viewer.html` in a browser.

> **Note:** The simulation uses an earlier set of platform dimensions in **inches** (`lp=7.125, l1=6.20, l2=4.50, lb=4.00`). The physical build uses different dimensions in **cm** (see below).

---

## Hardware (`Hardware/`)

A single Flask script, `balancer.py`, runs tracking, control and the tuning UI on the Pi.

### Components

| Part | Details |
|---|---|
| Controller | Raspberry Pi 4 (4 GB) |
| Camera | Raspberry Pi Camera v1.3 |
| Servo driver | PCA9685 (channels 14, 15, 13) |
| Actuators | 3 × 180° servos |
| Ball | 2.8 cm diameter |

**Platform geometry (cm):** LP = 7.62, L1 = 8.5, L2 = 4.5, LB = 4.0

### Web UI

Live camera feed with tracking overlay, plus three modes:

- **Manual**: drive each servo directly and set the platform height
- **Auto**: closed-loop PID balancing, with live Kp / Ki / Kd and α-β filter tuning, live ball position and velocity, and a max tilt limit up to 30° shown beside the geometric limit
- **Calibration**: servo offsets, hard min/max angle limits, platform centre, target point, work height, and ball colour tracking

**Extras**
- **Auto Level** to level the platform in one click
- **Tuning presets**: save, load and overwrite named presets. Each preset stores the PID gains and the ball colour/size settings, so there's one preset per ball
- **Target paths**: the ball follows a circle, square, triangle, star, line or spiral

**Run** (on the Pi)
```bash
cd Hardware
pip install -r requirements.txt
python balancer.py
```
Then open `http://<pi-ip-address>:5000` from any device on the same network.

---

## Acknowledgements

Hardware control code was developed using [signag/Ball-Balancing-Robot_CS](https://github.com/signag/Ball-Balancing-Robot_CS) as a reference.

## License

MIT. See [LICENSE](LICENSE).
