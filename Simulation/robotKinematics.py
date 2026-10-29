import math
import numpy as np


class RobotKinematics:
    """Implements inverse kinematics for a 3-legged parallel robot platform with ball joints.
    
    The robot consists of:
    - A top platform (radius lp) with 3 ball joints (A1, A2, A3)
    - 3 legs, each with an upper arm (length l1) and lower arm (length l2)
    - A bottom platform (radius lb) with 3 fixed base points (B1, B2, B3)
    - Middle node positions (C1, C2, C3) connecting the arms
    """

    def __init__(self, lp=7.125, l1=6.20, l2=4.50, lb=4.00, invert=False):
        """Initialize robot kinematics with geometric parameters.
        
        Args:
            lp: Radius of top platform
            l1: Length of upper arm
            l2: Length of lower arm
            lb: Radius of bottom platform
            invert: If True, arms bend inward; if False, arms bend outward
        """
        self.lp = lp    # Radius of top platform
        self.l1 = l1    # Upper arm length
        self.l2 = l2    # Lower arm length
        self.lb = lb    # Radius of bottom platform
        self.invert = invert    # Arm orientation: inward (True) or outward (False)

        # Compute workspace limits: max and min heights the top platform can reach
        self.maxh = self.compute_maxh() - 0.2    # Maximum height with safety margin
        self.minh = self.compute_minh() + 0.45   # Minimum height with safety margin
        self.p = [0.0, 0.0, self.maxh]    # Center position of top platform
        self.h = (self.maxh + self.minh) / 2     # Current height (initialized to midpoint)
        self.maxtheta = 10    # Maximum platform tilt angle in degrees (updated dynamically)
        
        # Ball joint positions on top platform (computed from orientation)
        self.A1 = [0, 0, 0]    # Joint 1 on top platform
        self.A2 = [0, 0, 0]    # Joint 2 on top platform
        self.A3 = [0, 0, 0]    # Joint 3 on top platform

        # Fixed base joint positions on bottom platform (120° apart, stationary)
        self.B1 = [0, 0, 0]    # Base joint 1 at 0°
        self.B2 = [0, 0, 0]    # Base joint 2 at 120°
        self.B3 = [0, 0, 0]    # Base joint 3 at 240°

        # Elbow/middle node positions connecting upper and lower arms
        self.C1 = [0.0, 0.0, 0.0]    # Elbow position for leg 1
        self.C2 = [0.0, 0.0, 0.0]    # Elbow position for leg 2
        self.C3 = [0.0, 0.0, 0.0]    # Elbow position for leg 3

        # Calculate maximum valid tilt angle for current height
        self.max_theta(self.h)

        # Joint angles (in radians) for each leg
        self.theta1 = 0    # Angle of leg 1
        self.theta2 = 0    # Angle of leg 2
        self.theta3 = 0    # Angle of leg 3

    def compute_maxh(self):
        """Compute maximum height when top platform is parallel to bottom platform.
        
        Uses the constraint that at max height, the fully extended arms (l1+l2) form
        the hypotenuse of a right triangle, with the horizontal distance being (lp-lb).
        """
        return math.sqrt(((self.l1 + self.l2) ** 2) - ((self.lp - self.lb) ** 2))

    def compute_minh(self):
        """Compute minimum height when top platform is parallel to bottom platform.
        
        The minimum height depends on which arm is longer. Uses geometric constraints
        to find the configuration where the arms are most compressed.
        """
        if self.l1 > self.l2:
            # Upper arm longer: min height uses l1 as hypotenuse
            return math.sqrt((self.l1 ** 2) - ((self.lb + self.l2 - self.lp) ** 2))
        elif self.l2 > self.l1:
            # Lower arm longer: min height uses arm difference as hypotenuse
            return math.sqrt(((self.l2 - self.l1) ** 2) - ((self.lp - self.lb) ** 2))
        else:
            # Equal arm lengths
            return 0
    
    def solve_top(self, a, b, c, h):
        """Compute positions of ball joints on top platform given its orientation and height.
        
        Args:
            a, b, c: Components of platform orientation unit vector [a, b, c]
            h: Height of platform center
            
        The three ball joints (A1, A2, A3) are positioned 120° apart on the platform
        at radius lp from the center, projected onto the tilted platform surface.
        A2 is positioned on the y=0 plane for reference.
        """
        if not self.invert:
            # Normal arm configuration: arms bend away from platform center
            # A1, A2, A3 are ball-joint coordinates on top platform at 120° intervals
            # A2 is the reference joint on plane y=0
            self.A1 = [ -(self.lp*c) / (math.sqrt(4*c**2 + (a - math.sqrt(3)*b)**2)),
                (math.sqrt(3)*self.lp*c) / (math.sqrt(4*c**2 + (a - math.sqrt(3)*b)**2)),
                h + ((a - math.sqrt(3)*b)*self.lp) / (math.sqrt(4*c**2 + (a - math.sqrt(3)*b)**2))]
            
            # Joint 2 positioned on y=0 plane
            self.A2 = [ (self.lp*c) / (math.sqrt(c**2 + a**2)),
                    0,
                        h - ((self.lp*a) / (math.sqrt(c**2 + a**2)))]
            
            # Joint 3 at 240° from joint 1
            self.A3 = [ -(self.lp*c) / (math.sqrt(4*c**2 + (a + math.sqrt(3)*b)**2)),
                -(math.sqrt(3)*self.lp*c) / (math.sqrt(4*c**2 + (a + math.sqrt(3)*b)**2)),
                h + ((a + math.sqrt(3)*b)*self.lp) / (math.sqrt(4*c**2 + (a + math.sqrt(3)*b)**2))]
        else:
            # Inverted arm configuration: arms bend toward platform center
            self.A1 = [ -(self.lp*c) / (math.sqrt(4*c**2 + (a - math.sqrt(3)*b)**2)),
                (math.sqrt(3)*self.lp*c) / (math.sqrt(4*c**2 + (a - math.sqrt(3)*b)**2)),
                h + ((a - math.sqrt(3)*b)*self.lp) / (math.sqrt(4*c**2 + (a - math.sqrt(3)*b)**2))]
            
            self.A2 = [ (self.lp*c) / (math.sqrt(c**2 + a**2)),
                    0,
                        h - ((self.lp*a) / (math.sqrt(c**2 + a**2)))]
            
            self.A3 = [ -(self.lp*c) / (math.sqrt(4*c**2 + (a + math.sqrt(3)*b)**2)),
                -(math.sqrt(3)*self.lp*c) / (math.sqrt(4*c**2 + (a + math.sqrt(3)*b)**2)),
                h + ((a + math.sqrt(3)*b)*self.lp) / (math.sqrt(4*c**2 + (a + math.sqrt(3)*b)**2))]


    def solve_middle(self):
        """Compute elbow positions (middle nodes C1, C2, C3) that connect arms to platform joints.
        
        Given the ball joint positions on the top platform (A1, A2, A3) and fixed base
        positions (B1, B2, B3), solves for the middle nodes (elbows) such that:
        - Distance from Ai to Ci equals l1 (upper arm length)
        - Distance from Ci to Bi equals l2 (lower arm length)
        """
        # Extract coordinates of top platform ball joints
        a11, a12, a13 = map(float, self.A1)
        a21, a22, a23 = map(float, self.A2)
        a31, a32, a33 = map(float, self.A3)


        # Solve for elbow position C1 (leg 1)
        # Uses quadratic formula coefficients from geometric constraints
        # Equations derive from distance constraints: |A1-C1|=l1 and |C1-B1|=l2
        p1 = (-a11 + math.sqrt(3)*a12 - 2*self.lb) / a13
        q1 = (a11**2 + a12**2 + a13**2 + self.l2**2 - self.l1**2 - self.lb**2) / (2*a13)
        r1 = p1**2 + 4
        s1 = 2*p1*q1 + 4*self.lb
        t1 = q1**2 + self.lb**2 - self.l2**2

        # Solve for elbow position C2 (leg 2, on y=0 plane)
        p2 = (self.lb - a21) / a23
        q2 = (a21**2 + a23**2 - self.lb**2 + self.l2**2 - self.l1**2) / (2*a23)
        r2 = p2**2 + 1
        s2 = 2*(p2*q2 - self.lb)
        t2 = q2**2 - self.l2**2 + self.lb**2

        # Solve for elbow position C3 (leg 3)
        p3 = (-a31 - math.sqrt(3)*a32 - 2*self.lb) / a33
        q3 = (a31**2 + a32**2 + a33**2 + self.l2**2 - self.l1**2 - self.lb**2) / (2*a33)
        r3 = p3**2 + 4
        s3 = 2*p3*q3 + 4*self.lb
        t3 = q3**2 + self.lb**2 - self.l2**2

        if not self.invert:
            # Normal configuration: elbows above base (positive z)
            # Solve quadratic for C1 elbow x-coordinate, then compute y, z
            c11 = (-s1 - math.sqrt(s1**2 - 4*r1*t1)) / (2*r1)
            c12 = -math.sqrt(3) * c11
            c13 = math.sqrt(self.l2**2 - 4*(c11**2) - 4*self.lb*c11 - self.lb**2)
            self.C1 = [c11, c12, c13]

            # Solve for C2 elbow on y=0 plane
            c21 = (-s2 + math.sqrt(s2**2 - 4*r2*t2)) / (2*r2)
            c22 = 0
            c23 = math.sqrt(self.l2**2 - (c21 - self.lb)**2)
            self.C2 = [c21, c22, c23]

            # Solve for C3 elbow at 240° from C1
            c31 = (-s3 - math.sqrt(s3**2 - 4*r3*t3)) / (2*r3)
            c32 = math.sqrt(3) * c31
            c33 = math.sqrt(self.l2**2 - 4*(c31**2) - 4*self.lb*c31 - self.lb**2)
            self.C3 = [c31, c32, c33]

        else:
            # Inverted configuration: elbows below base (negative z)
            # Same x, y coordinates as normal but negated z
            c11 = (-s1 - math.sqrt(s1**2 - 4*r1*t1)) / (2*r1)
            c12 = -math.sqrt(3) * c11
            c13 = math.sqrt(self.l2**2 - 4*(c11**2) - 4*self.lb*c11 - self.lb**2)
            self.C1 = [c11, c12, -c13]

            c21 = (-s2 + math.sqrt(s2**2 - 4*r2*t2)) / (2*r2)
            c22 = 0
            c23 = math.sqrt(self.l2**2 - (c21 - self.lb)**2)
            self.C2 = [c21, c22, -c23]

            c31 = (-s3 - math.sqrt(s3**2 - 4*r3*t3)) / (2*r3)
            c32 = math.sqrt(3) * c31
            c33 = math.sqrt(self.l2**2 - 4*(c31**2) - 4*self.lb*c31 - self.lb**2)
            self.C3 = [c31, c32, -c33]

    def solve_inverse_kinematics_vector(self, a, b, c, h):
        """Solve inverse kinematics given platform orientation vector and height.
        
        Args:
            a, b, c: Normalized orientation vector components [a, b, c]
            h: Height of platform center
            
        Computes the required joint angles (theta1, theta2, theta3) to achieve the
        desired platform position and orientation.
        """
        # Set fixed base joint positions at 120° intervals around origin
        self.B1 = [-0.5*self.lb, math.sqrt(3)*0.5*self.lb, 0]    # 120° position
        self.B2 = [self.lb, 0, 0]                                # 0° position
        self.B3 = [-0.5*self.lb, -1*math.sqrt(3)*0.5*self.lb, 0] # 240° position

        # Compute top platform joint positions given orientation and height
        self.solve_top(a, b, c, h)
        # Compute elbow positions given top joints and base joints
        self.solve_middle()

        # Calculate joint angles from computed elbow positions
        # theta represents the angle each leg makes relative to vertical
        self.theta1 = math.pi/2 - math.atan2(math.sqrt(self.C1[0]**2 + self.C1[1]**2) - self.lb, self.C1[2])
        self.theta2 = math.atan2(self.C2[2], self.C2[0] - self.lb)
        self.theta3 = math.pi/2 - math.atan2(math.sqrt(self.C3[0]**2 + self.C3[1]**2) - self.lb, self.C3[2])

    def solve_inverse_kinematics_spherical(self, theta, phi, h):
        """Solve inverse kinematics given platform tilt in spherical coordinates.
        
        Args:
            theta: Polar angle (tilt angle from vertical) in degrees
            phi: Azimuthal angle (rotation around vertical axis) in degrees
            h: Height of platform center
            
        Converts spherical coordinates to orientation vector and solves IK.
        Clamps theta to maximum valid angle for given height.
        """
        # Store current height and compute maximum valid tilt angle
        self.h = h
        self.max_theta(h)

        # Clamp requested tilt angle to maximum achievable for current height
        theta = min(theta, self.maxtheta)

        # Convert spherical to Cartesian orientation vector
        a = math.sin(math.radians(theta)) * math.cos(math.radians(phi))
        b = math.sin(math.radians(theta)) * math.sin(math.radians(phi))
        c = math.cos(math.radians(theta))
        
        try:
            # Solve IK using the converted orientation vector
            self.solve_inverse_kinematics_vector(a, b, c, h)
        except Exception as e:
            # Log problematic inputs for debugging if IK solve fails
            print(a, b, c, h, theta, phi)
            pass


    def max_theta(self, h, tol=1e-3):
        """Find maximum valid tilt angle for the given platform height using binary search.
        
        Args:
            h: Current platform height
            tol: Tolerance for binary search convergence (radians)
            
        Uses binary search to find the largest tilt angle where valid arm configurations
        exist. A configuration is valid if both legs can reach from platform joints to
        base joints while respecting arm length constraints.
        """
        theta_low, theta_high = 0.0, math.radians(20)
        
        def valid(theta):
            """Check if a tilt angle produces geometrically valid arm configurations."""
            c = math.cos(theta)
            # Test both positive and negative tilt directions
            for s in (1, -1):
                # Compute position of joint A2 (reference joint on y=0 plane) at this tilt
                a21 = self.lp * c
                a23 = h - self.lp * (s * math.sin(theta))
                try:
                    # Check if elbow position C2 can be computed and arm constraints satisfied
                    p2 = (self.lb - a21) / a23
                    q2 = (a21**2 + a23**2 - self.lb**2 + self.l2**2 - self.l1**2) / (2 * a23)
                    r2 = p2**2 + 1
                    s2 = 2 * (p2 * q2 - self.lb)
                    t2 = q2**2 - self.l2**2 + self.lb**2
                    # Check discriminant for valid elbow position
                    disc = s2**2 - 4 * r2 * t2
                    if disc < 0: return False
                    c21 = (-s2 + math.sqrt(disc)) / (2 * r2)
                    # Check if elbow z-coordinate is real
                    delta = self.l2**2 - (c21 - self.lb)**2
                    if delta < 0: return False
                    c23 = math.sqrt(delta)
                    # Verify computed elbow satisfies arm length constraints
                    if abs(math.sqrt((a21-c21)**2 + (a23-c23)**2) - self.l1) > 1e-3: return False
                    if abs(math.sqrt((self.lb-c21)**2 + c23**2) - self.l2) > 1e-3: return False
                except:
                    return False
            return True
        
        # Binary search for maximum valid tilt angle
        while theta_high - theta_low > tol:
            theta_mid = (theta_low + theta_high) / 2
            if valid(theta_mid): 
                theta_low = theta_mid  # Can tilt more
            else: 
                theta_high = theta_mid  # Tilt too much

        # Convert result to degrees and apply safety margin
        self.maxtheta = max(0, math.degrees(round(theta_low, 4)) - 0.5)




    