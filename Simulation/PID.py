"""
PID Controller for Ball Balancing Robot
Controls platform tilt to balance a ball at a target position using PID feedback control.
"""

import math
import time

class PIDcontroller:
    """
    Same PID structure as the original PID.py, with ONE change: the
    derivative term now uses REAL velocity passed in directly, instead of
    finite-differencing consecutive position readings.

    Why: the original design (d_err = (err - prev_err) / dt) is a
    reasonable estimate when velocity isn't otherwise available (e.g. from
    noisy camera position readings alone). But our simulation already knows
    the ball's true velocity exactly -- feeding that in directly is more
    accurate and removes one source of noise/lag from the loop. When this
    gets connected to the real camera later, you have a choice: keep this
    velocity-input version if you add a velocity estimator (e.g. a Kalman
    filter) to your vision pipeline, or fall back to the original
    finite-difference version if you only ever have raw position.
    """
    def __init__(self, kp, ki, kd, alpha, beta, max_theta, conversion="linear"):
        # PID gain coefficients
        self.kp, self.ki, self.kd = kp, ki, kd

        # alpha: Exponential filter coefficient (0-1) for smoothing output
        # beta: Scaling coefficient for converting magnitude (linear or tanh)
        self.alpha = alpha  # Exponential filter: alpha*x + (1-alpha)*x_last
        self.beta = beta    # Coefficient for converting magnitude, beta*x or tanh(beta*x)
        # Maximum tilt angle in degrees
        self.max_theta = max_theta

        # Select magnitude conversion method: linear (1) or tanh (0)
        if conversion == "linear":
            self.magnitude_convert = 1
        elif conversion == "tanh":
            self.magnitude_convert = 0
        else:
            self.magnitude_convert = -1

        # Filtered output from previous iteration (Cartesian coordinates)
        self.prev_out_x = 0.0
        self.prev_out_y = 0.0

        # Integral accumulator for error terms
        self.sum_err_x = 0.0  # Integral
        self.sum_err_y = 0.0  # Integral

        # Timestamp for calculating delta time between iterations
        self.last_time = None

    def pid(self, target, current, velocity):
        """
        Calculate PID control output to move ball toward target.
        
        Args:
            target:   (x_t, y_t)  desired position
            current:  (x, y)      current ball position
            velocity: (vx, vy)    current ball velocity -- used directly for
                                   the derivative term instead of estimating it
        
        Returns:
            theta: tilt angle magnitude in degrees
            phi:   tilt direction in degrees (0-360)
        """
        # Calculate time step for integral and derivative calculations
        new_time = time.perf_counter()
        dt = new_time - self.last_time if self.last_time is not None else 0.001

        # Calculate position error (current minus target)
        err_x = current[0] - target[0]
        err_y = current[1] - target[1]
        # Accumulate error for integral term
        self.sum_err_x += err_x * dt
        self.sum_err_y += err_y * dt

        # Derivative term: use actual velocity instead of finite difference
        # d/dt(current - target) = d/dt(current) since target is constant --
        # that's exactly the ball's velocity, so use it directly.
        d_err_x, d_err_y = velocity[0], velocity[1]

        # Combine proportional, integral, and derivative terms
        pid_x = self.kp * err_x + self.ki * self.sum_err_x + self.kd * d_err_x
        pid_y = self.kp * err_y + self.ki * self.sum_err_y + self.kd * d_err_y

        # Sign flip: err points FROM target TO the ball, so tilting the plate
        # "toward" +pid_x/+pid_y actually tilts that side DOWN which pushes the ball 
        # further from center, not back to it. 
        # Negating here makes the plate tilt the OPPOSITE
        # i.e. towards the target
        
        pid_x, pid_y = -pid_x, -pid_y

        # Apply exponential low-pass filter to smooth output and reduce jitter
        filtered_x = self.alpha * pid_x + (1 - self.alpha) * self.prev_out_x
        filtered_y = self.alpha * pid_y + (1 - self.alpha) * self.prev_out_y

        # Convert from Cartesian (x, y) to spherical (theta, phi) coordinates
        # phi: direction angle in degrees (0-360)
        phi = math.degrees(math.atan2(filtered_y, filtered_x))
        if phi < 0:
            phi += 360
        
        # r: magnitude in Cartesian coordinates
        r = math.sqrt(filtered_x**2 + filtered_y**2)
        
        # Convert magnitude to tilt angle based on selected conversion method
        if self.magnitude_convert == 1:
            # Linear: directly scale by beta, clamped between 0 and max_theta
            theta = min(max(0, self.beta * r), self.max_theta)
        else:
            # Tanh: hyperbolic tangent scaling for smoother response
            theta = max(0, 15 * math.tanh(self.beta * r))

        # Update state for next iteration
        self.prev_out_x = filtered_x
        self.prev_out_y = filtered_y
        self.last_time = new_time

        return theta, phi  # degrees
