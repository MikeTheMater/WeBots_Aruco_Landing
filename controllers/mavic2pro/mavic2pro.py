from controller import Robot, Supervisor
import sys
import numpy as np
import cv2
import math
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import time

def clamp(value, value_min, value_max):
    return min(max(value, value_min), value_max)

class Mavic(Robot):
    # Constants, empirically found.
    K_VERTICAL_THRUST = 68.5  # with this thrust, the drone lifts.
    K_VERTICAL_OFFSET = 0.6   # Vertical offset where the robot actually targets to stabilize itself.
    K_VERTICAL_P = 2.3        # P constant of the vertical PID.
    K_ROLL_P = 50.0           # P constant of the roll PID.
    K_PITCH_P = 35.0          # P constant of the pitch PID.
    MAX_YAW_DISTURBANCE = 0.8
    MAX_PITCH_DISTURBANCE = -5.0
    target_precision = 0.15    # Precision between the target position and the robot position in meters

    def __init__(self):
        Robot.__init__(self)
        self.time_step = int(self.getBasicTimeStep())
        # Initialize devices.
        self.init_devices()
        self.init_motors()
        #self.init_bounding_box()
        self.current_pose = [0, 0, 0, 0, 0, 0]  # X, Y, Z, yaw, pitch, roll
        self.target_position = [0, 0, 0]
        self.target_altitude = 5
        self.sinusoidal_path_amplitude = 20  # Increased amplitude
        self.sinusoidal_path_frequency = 0.5
        self.marker_detected = False
        self.marker_position = [0, 0, 0]
        self.target_index = 0
        self.landed = False
        
        self.avoiding = False
        self.avoid_goal_xy = [0.0, 0.0]
        self.avoid_goal_alt = None
       
        self.avoid_reach_eps = 0.4  # m, how close before we clear avoidance
        
        # --- altitude/vertical control ---
        self.K_VERTICAL_D = 3.8        # derivative gain on vertical speed (tune 2.0–5.0)
        self.alt_index = 2             # assume Z-up first; we auto-switch if wrong
        self._alt_prev = None          # last altitude sample
        self.alt_slew_rate = 0.08      # faster slew so changes are visible during avoidance

        # --- avoidance instrumentation (no spam) ---
        self._avoid_id = 0
        self._avoid_started_t = 0.0
        self._alt_at_avoid_start = None
        self._other_alt_start = None



    def init_devices(self):
        self.camera = self.getDevice("camera")
        self.camera.enable(self.time_step)
        self.camera_pitch_motor = self.getDevice("camera pitch")
        self.camera_pitch_motor.setPosition(1.7)  # Orient the camera downwards
        self.imu = self.getDevice("inertial unit")
        self.imu.enable(self.time_step)
        self.gps = self.getDevice("gps")
        self.gps.enable(self.time_step)
        self.gyro = self.getDevice("gyro")
        self.gyro.enable(self.time_step)
        self.emitter = self.getDevice("emitter")
        self.receiver = self.getDevice("receiver")
        self.receiver.enable(self.time_step)
        
        self.max_fwd_cmd_cruise = 1.0   # how “fast” forward (disturbance units)
        self.max_fwd_cmd_avoid  = 0.25  # slow creep during avoidance
        self.pitch_cmd_rate     = 0.08  # max change of pitch cmd per step
        self.yaw_boost_avoid    = 1.3   # turn a bit harder during avoidance
        self.avoid_offset       = 1.5   # meters to sidestep right


    def init_motors(self):
        self.front_left_motor = self.getDevice("front left propeller")
        self.front_right_motor = self.getDevice("front right propeller")
        self.rear_left_motor = self.getDevice("rear left propeller")
        self.rear_right_motor = self.getDevice("rear right propeller")
        self.motors = [self.front_left_motor, self.front_right_motor,
                       self.rear_left_motor, self.rear_right_motor]
        for motor in self.motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(1)
    
    def set_position(self, pos):
        self.current_pose = pos
        
    def set_id(self, id):
        self.id = id

    def update_sinusoidal_waypoints(self, current_time):
        x = current_time * self.sinusoidal_path_frequency
        y = self.sinusoidal_path_amplitude * np.sin(x)
        self.target_position = [x, y]

    def update_linear_waypoints(self, waypoints):
        if all([abs(x1 - x2) < self.target_precision for (x1, x2) in zip(self.target_position[0:2], self.current_pose[0:2])]):
            self.target_index += 1
            if self.target_index > len(waypoints) - 1:
                self.target_index = 0
        self.target_position[0:2] = waypoints[self.target_index]
    
    def compute_movement(self):
        # Calculate the yaw and pitch disturbances to navigate towards the target position.
        target_yaw = np.arctan2(
            self.target_position[1] - self.current_pose[1], 
            self.target_position[0] - self.current_pose[0]
        )

        angle_to_target = target_yaw - self.current_pose[3]
        angle_to_target = (angle_to_target + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]

        yaw_disturbance = self.MAX_YAW_DISTURBANCE * angle_to_target / np.pi
        distance_to_target = np.sqrt(
            (self.target_position[0] - self.current_pose[0]) ** 2 + 
            (self.target_position[1] - self.current_pose[1]) ** 2
        )
        k_dist = 1.0  # forward demand per meter
        pitch_disturbance = clamp(-k_dist * distance_to_target, self.MAX_PITCH_DISTURBANCE, 0.0)
        # pitch_disturbance = clamp(
        #     self.MAX_PITCH_DISTURBANCE * distance_to_target / 10, 
        #     self.MAX_PITCH_DISTURBANCE, 0
        # )

        return yaw_disturbance, pitch_disturbance
        
    def calculate_translation_rel_to_world(self, translation):
        return [translation[0] + self.gps.getValues()[0], translation[1] + self.gps.getValues()[1], translation[2] + self.gps.getValues()[2]]
    
    def detect_aruco_marker(self):
        image = self.camera.getImage()
        height, width = self.camera.getHeight(), self.camera.getWidth()
        image_array = np.frombuffer(image, dtype=np.uint8).reshape((height, width, 4))
        gray_image = cv2.cvtColor(image_array[:, :, :3], cv2.COLOR_BGR2GRAY)

        # Save image for debugging
        #cv2.imwrite("drone_view.jpg", gray_image)

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray_image, aruco_dict, parameters=parameters)

        if ids is not None and self.id in ids:
            index = list(ids).index(self.id)
            marker_center = np.mean(corners[index][0], axis=0)

            # Convert pixel coordinates to world coordinates
            x_world, y_world = self.pixel_to_world(marker_center[0], marker_center[1])
            self.marker_position = [x_world, y_world, 0]
            return True  # Marker with the specified ID is detected
        return False

    def pixel_to_world(self, x_pixel, y_pixel):
        # Assuming camera is pointing downwards and altitude is a good approximation of distance to ground
        altitude = self.gps.getValues()[2]
        image_width, image_height = self.camera.getWidth(), self.camera.getHeight()

        # Approximate field of view of the camera
        camera_fov_horizontal = 0.785 
        camera_fov_vertical = 0.785 

        # Calculate real world distance per pixel at current altitude
        real_world_per_pixel_x = 2 * altitude * np.tan(camera_fov_horizontal / 2) / image_width
        real_world_per_pixel_y = 2 * altitude * np.tan(camera_fov_vertical / 2) / image_height

        # Adjust based on trial and error calibration
        calibration_factor_x = -0.1 # Adjust based on your trials
        calibration_factor_y = -0.2  # Adjust based on your trials
        real_world_per_pixel_x *= calibration_factor_x
        real_world_per_pixel_y *= calibration_factor_y

        # Convert to real world coordinates relative to drone
        x_world_relative = (x_pixel - image_width / 2) * real_world_per_pixel_x
        y_world_relative = (y_pixel - image_height / 2) * real_world_per_pixel_y

        # Convert to absolute world coordinates
        drone_x, drone_y, _ = self.gps.getValues()
        x_world = drone_x + x_world_relative
        y_world = drone_y + y_world_relative

        return x_world, y_world

    def land(self,starting_altitude):
        # Gradual descent parameters
        descent_rate = 0.01  # Meters per time step
        touchdown_altitude = 0.2  # Altitude at which to cut off motors

        starting_altitude=int(starting_altitude)
        while self.step(self.time_step) != -1:
            _, _, altitude = self.gps.getValues()
            roll, pitch, yaw = self.imu.getRollPitchYaw()


            if altitude <= touchdown_altitude:
                # If the drone is close enough to the ground, cut off the motors
                for motor in self.motors:
                    motor.setVelocity(0.0)
                break
            
            yaw_disturbance, pitch_disturbance = self.move_to_target([self.marker_position])
            
            # Gradually reduce the altitude
            self.target_altitude -= descent_rate
            self.target_altitude = max(self.target_altitude, 0)  # Ensure it doesn't go below zero

            # Update motor speeds for controlled descent
            roll, pitch, yaw = self.imu.getRollPitchYaw()
            roll_acceleration, pitch_acceleration, _ = self.gyro.getValues()
            roll_input = self.K_ROLL_P * clamp(roll, -1, 1) + roll_acceleration
            pitch_input = self.K_PITCH_P * clamp(pitch, -1, 1) + pitch_acceleration + pitch_disturbance
            yaw_input = yaw_disturbance  # No yaw adjustment needed during landing
            clamped_difference_altitude = clamp(self.target_altitude - altitude + self.K_VERTICAL_OFFSET, -1, 1)
            vertical_input = self.K_VERTICAL_P * pow(clamped_difference_altitude, 3.0)

            front_left_motor_input = self.K_VERTICAL_THRUST + vertical_input - yaw_input + pitch_input - roll_input
            front_right_motor_input = self.K_VERTICAL_THRUST + vertical_input + yaw_input + pitch_input + roll_input
            rear_left_motor_input = self.K_VERTICAL_THRUST + vertical_input + yaw_input - pitch_input - roll_input
            rear_right_motor_input = self.K_VERTICAL_THRUST + vertical_input - yaw_input - pitch_input + roll_input

            if math.isnan(front_left_motor_input) or math.isnan(front_right_motor_input) or math.isnan(rear_left_motor_input) or math.isnan(rear_right_motor_input):
                print("NaN value detected. Changing them to 0.")
                for motor in self.motors:
                    motor.setVelocity(0.0)
                break
            
            self.front_left_motor.setVelocity(front_left_motor_input)
            self.front_right_motor.setVelocity(-front_right_motor_input)
            self.rear_left_motor.setVelocity(-rear_left_motor_input)
            self.rear_right_motor.setVelocity(rear_right_motor_input)
            
            self.detect_aruco_marker()
            #print("Marker position: ", self.marker_position)
        self.camera.disable()
        print("Landing completed.")
        self.landed=True
        self.setCustomData("landed")

    def move_to_target(self, waypoints, verbose_movement=False, verbose_target=True):
        """
        Move the robot to the given coordinates
        Parameters:
            waypoints (list): list of X,Y coordinates
            verbose_movement (bool): whether to print remaning angle and distance or not
            verbose_target (bool): whether to print targets or not
        Returns:
            yaw_disturbance (float): yaw disturbance (negative value to go on the right)
            pitch_disturbance (float): pitch disturbance (negative value to go forward)
        """

        if self.target_position[0:2] == [0, 0]:  # Initialization
            self.target_position[0:2] = waypoints[0]
            self.target_altitude = self.random_altitude[self.alt_counter]
            self.alt_counter+=1
            if verbose_target:
                print("First target: ", self.target_position[0:2])
                print("Target altitude: ", self.target_altitude)

        # if the robot is at the position with a precision of target_precision
        if all([abs(x1 - x2) < self.target_precision for (x1, x2) in zip(self.target_position, self.current_pose[0:2])]):

            self.target_index += 1
            if self.target_index > len(waypoints) - 1:
                self.target_index = 0
            self.target_position[0:2] = waypoints[self.target_index]
            if verbose_target:
                print("Target reached! New target: ",
                      self.target_position[0:2])

        # This will be in ]-pi;pi]
        self.target_position[2] = np.arctan2(
            self.target_position[1] - self.current_pose[1], self.target_position[0] - self.current_pose[0])
        # This is now in ]-2pi;2pi[
        angle_left = self.target_position[2] - self.current_pose[3]
        # Normalize turn angle to ]-pi;pi]
        angle_left = (angle_left + 2 * np.pi) % (2 * np.pi)
        if (angle_left > np.pi):
            angle_left -= 2 * np.pi

        # Turn the robot to the left or to the right according the value and the sign of angle_left
        yaw_disturbance = self.MAX_YAW_DISTURBANCE * angle_left / (2 * np.pi)
        # non proportional and decreasing function
        pitch_disturbance = clamp(
            np.log10(abs(angle_left)), self.MAX_PITCH_DISTURBANCE, 0.1)

        if verbose_movement:
            distance_left = np.sqrt(((self.target_position[0] - self.current_pose[0]) ** 2) + (
                (self.target_position[1] - self.current_pose[1]) ** 2))
            print("remaning angle: {:.4f}, remaning distance: {:.4f}".format(
                angle_left, distance_left))
        return yaw_disturbance, pitch_disturbance
    
    def move_right_by_motor_control(self, speed_difference, target_altitude):
        # Constants (adjust as needed)
        k_roll_p = self.K_ROLL_P
        k_pitch_p = self.K_PITCH_P
        k_vertical_p = self.K_VERTICAL_P
        k_vertical_thrust = self.K_VERTICAL_THRUST
        k_vertical_offset = self.K_VERTICAL_OFFSET

        # Get current sensor values
        roll = self.imu.getRollPitchYaw()[0]
        pitch = self.imu.getRollPitchYaw()[1]
        altitude = self.gps.getValues()[2]
        roll_velocity = self.gyro.getValues()[0]
        pitch_velocity = self.gyro.getValues()[1]
        pitch_disturbance = 0
        yaw_disturbance = 0     # ← keep zero; don't steer to “move right”
        roll_disturbance = -clamp(speed_difference, 0.0, 1.0)  # scale 0..1

        roll_input  = k_roll_p  * clamp(roll, -1.0, 1.0)  + roll_velocity  + roll_disturbance
        pitch_input = k_pitch_p * clamp(pitch, -1.0, 1.0) + pitch_velocity + pitch_disturbance
        yaw_input   = 0.0  # ← no yaw for lateral sidestep

        clamped_difference_altitude = clamp(target_altitude - altitude + k_vertical_offset, -1.0, 1.0)
        vertical_input = k_vertical_p * pow(clamped_difference_altitude, 3.0)

        if abs(roll) > 1.0 or abs(pitch) > 1.0:  # ~57°
            yaw_input = 0.0
            pitch_input = 0.0
            roll_input = 0.0
            # slightly increase vertical to stabilize
        
        # Actuate the motors considering all the computed inputs
        front_left_motor_input = (k_vertical_thrust + vertical_input - roll_input + pitch_input - yaw_input)
        front_right_motor_input = (k_vertical_thrust + vertical_input + roll_input + pitch_input + yaw_input)
        rear_left_motor_input = (k_vertical_thrust + vertical_input - roll_input - pitch_input + yaw_input)
        rear_right_motor_input = (k_vertical_thrust + vertical_input + roll_input - pitch_input - yaw_input)

        # Set the motor velocities
        self.front_left_motor.setVelocity(front_left_motor_input)
        self.front_right_motor.setVelocity(-front_right_motor_input)
        self.rear_left_motor.setVelocity(-rear_left_motor_input)
        self.rear_right_motor.setVelocity(rear_right_motor_input)

    def _start_avoidance(self, suggested_alt):
        yaw = self.current_pose[3]
        right = np.array([-math.sin(yaw), math.cos(yaw)])
        offset = 1.5 * right  # a bit wider sidestep helps
        self.avoid_goal_xy = [self.current_pose[0] + offset[0],
                            self.current_pose[1] + offset[1]]
        self.avoid_goal_alt = suggested_alt
        self.avoiding = True

        # --- one-time bookkeeping (no spam) ---
        self._avoid_id += 1
        self._avoid_started_t = self.getTime()
        gpsv = self.gps.getValues()
        self._alt_at_avoid_start = gpsv[self.alt_index]
        self._other_alt_start = gpsv[1 if self.alt_index == 2 else 2]


    def _rate_limit(self, new, old, step):
        dv = new - old
        if dv > step:  return old + step
        if dv < -step: return old - step
        return new

        
    def _sat(self, v, lo=0.0, hi=400.0):  # tune hi to your model
        return max(lo, min(hi, v))

        
    def run(self):
        # --- patrol scaffolding (same as before / lightweight) ---
        waypoints = [np.random.randint(-2, 2, 2).tolist() for _ in range(50)]
        self.random_altitude = [round(np.random.uniform(10, 15), 1) for _ in range(50)]
        self.alt_counter = 0
        self.target_altitude = self.random_altitude[self.alt_counter]
        alt_count = 0
        last_time = 0

        # smoothing state
        if not hasattr(self, "_ys"):
            self._ys = 0.0   # smoothed yaw command
            self._ps = 0.0   # smoothed pitch command

        while self.step(self.time_step) != -1:
            # --- sensors ---
            roll, pitch, yaw = self.imu.getRollPitchYaw()
            gpsv = self.gps.getValues()
            x_pos, y_pos = gpsv[0], gpsv[1]
            # altitude will be selected later via self.alt_index (3)
            roll_accel, pitch_accel, _ = self.gyro.getValues()
            self.set_position([x_pos, y_pos, gpsv[2], yaw, pitch, roll])

            # --- defaults each frame ---
            yaw_disturbance = 0.0
            pitch_disturbance = 0.0

            # --- read supervisor flag robustly ---
            col = (self.getCustomData() or "0").strip()   # e.g. "0" or "nx ny nz"
            collision_active = (col != "0")

            nx = ny = 0.0
            nz = self.target_altitude
            if collision_active:
                try:
                    nx, ny, nz = map(float, col.split())
                except Exception:
                    nz = self.target_altitude  # malformed payload -> ignore altitude change

            # --- avoidance vs normal targeting ---
            if self.avoiding or collision_active:
                # Start avoidance once per event
                if collision_active and not self.avoiding:
                    # (4) Bias altitude if suggested change is too small (±0.8 m)
                    if abs(nz - self.target_altitude) < 0.2:
                        dz = 0.8 if (getattr(self, "id", 0) % 2 == 0) else -0.8
                        nz = self.target_altitude + dz
                    self._start_avoidance(nz)
                    # one-time info (no per-step spam)
                    try:
                        print(f"[AVOID#{self._avoid_id} id={getattr(self,'id','?')}] start alt_target={nz:.2f}")
                    except Exception:
                        pass

                # Fly to the sidestep waypoint (stable), slew altitude toward avoid_goal_alt
                self.target_position[0:2] = self.avoid_goal_xy
                if self.avoid_goal_alt is not None:
                    dz = self.avoid_goal_alt - self.target_altitude
                    if abs(dz) > self.alt_slew_rate:
                        self.target_altitude += self.alt_slew_rate * np.sign(dz)
                    else:
                        self.target_altitude = self.avoid_goal_alt

                # Compute heading/forward demand toward sidestep
                yaw_disturbance, pitch_disturbance = self.compute_movement()

                # If supervisor flag cleared, keep avoiding until we reach sidestep
                if not collision_active:
                    dx = self.current_pose[0] - self.avoid_goal_xy[0]
                    dy = self.current_pose[1] - self.avoid_goal_xy[1]
                    if math.hypot(dx, dy) < self.avoid_reach_eps:
                        self.avoiding = False
                        self.avoid_goal_alt = None

            else:
                # --- NORMAL / MARKER LOGIC ---
                if not self.marker_detected:
                    current_time = self.getTime()
                    if gpsv[self.alt_index] > self.target_altitude - 1:
                        if int(current_time) % 2 == 0 and alt_count < len(self.random_altitude) and int(current_time) != last_time:
                            self.target_altitude = self.random_altitude[alt_count]
                            self.target_position[0:2] = waypoints[alt_count]
                            alt_count += 1
                            last_time = int(current_time)
                        else:
                            self.update_sinusoidal_waypoints(current_time)
                        yaw_disturbance, pitch_disturbance = self.compute_movement()
                    else:
                        yaw_disturbance = 0.0
                        pitch_disturbance = 0.0
                else:
                    # Marker mode: fly to marker then land
                    self.target_position[0:2] = self.marker_position
                    yaw_disturbance, pitch_disturbance = self.move_to_target([self.marker_position])
                    if (abs(self.current_pose[0] - self.marker_position[0]) < 0.5 and
                        abs(self.current_pose[1] - self.marker_position[1]) < 0.5):
                        print("Landing on the marker.")
                        self.land(gpsv[self.alt_index])
                        break

                # Detect marker (no spam)
                if not self.marker_detected and self.detect_aruco_marker():
                    print("Marker detected, switching to marker mode.")
                    print("Marker position: ", self.marker_position)
                    self.marker_detected = True

            # === COMMON: smoothing (always runs) ===
            alpha = 0.35
            self._ys = (1.0 - alpha) * self._ys + alpha * yaw_disturbance
            raw_ps = (1.0 - alpha) * self._ps + alpha * pitch_disturbance
            self._ps = self._rate_limit(raw_ps, getattr(self, "_ps", 0.0), self.pitch_cmd_rate)

            # === (3) ALTITUDE + VERTICAL CONTROL WITH TILT COMPENSATION ===
            # Select altitude axis (assume self.alt_index is set in __init__, default 2)
            altitude = self.gps.getValues()[self.alt_index]
            dt = self.time_step / 1000.0
            if getattr(self, "_alt_prev", None) is None:
                vz = 0.0
            else:
                vz = (altitude - self._alt_prev) / dt
            self._alt_prev = altitude

            # Auto-switch altitude axis if avoidance started and our "altitude" doesn't move,
            # while the other axis does (Z<->Y swap). Check once ~0.8s after avoid start.
            if self.avoiding and (self.getTime() - getattr(self, "_avoid_started_t", 0.0)) > 0.8:
                alt_now = self.gps.getValues()[self.alt_index]
                other_idx = 1 if self.alt_index == 2 else 2
                other_now = self.gps.getValues()[other_idx]
                if (abs(alt_now - getattr(self, "_alt_at_avoid_start", alt_now)) < 0.05 and
                    abs(other_now - getattr(self, "_other_alt_start", other_now)) > 0.10):
                    self.alt_index = other_idx
                    self._alt_prev = None  # reset derivative
                    try:
                        print(f"[ROB:{getattr(self,'id','?')}] switched altitude axis to index {self.alt_index}")
                    except Exception:
                        pass
                    # refresh altitude with new axis
                    altitude = self.gps.getValues()[self.alt_index]

            # Vertical PD: P on altitude error + D on vertical speed
            alt_err = self.target_altitude - altitude
            alt_err = clamp(alt_err, -2.0, 2.0)
            vertical_input = self.K_VERTICAL_P * alt_err - getattr(self, "K_VERTICAL_D", 3.0) * vz

            # Tilt-compensated hover thrust
            tilt_comp = math.cos(roll) * math.cos(pitch)
            tilt_comp = max(0.5, tilt_comp)  # avoid division by tiny numbers
            hover_thrust = self.K_VERTICAL_THRUST / tilt_comp

            # === Motor mixing (always runs) ===
            roll_input  = self.K_ROLL_P  * clamp(roll,  -1.0, 1.0) + roll_accel
            pitch_input = self.K_PITCH_P * clamp(pitch, -1.0, 1.0) + pitch_accel + self._ps
            yaw_input   = self._ys

            fl = hover_thrust + vertical_input - yaw_input + pitch_input - roll_input
            fr = hover_thrust + vertical_input + yaw_input + pitch_input + roll_input
            rl = hover_thrust + vertical_input + yaw_input - pitch_input - roll_input
            rr = hover_thrust + vertical_input - yaw_input - pitch_input + roll_input

            # NaN guard
            if any(math.isnan(v) for v in (fl, fr, rl, rr)):
                for m in self.motors:
                    m.setVelocity(0.0)
                continue

            # Clamp + send
            fl = self._sat(fl); fr = self._sat(fr); rl = self._sat(rl); rr = self._sat(rr)
            self.front_left_motor.setVelocity(fl)
            self.front_right_motor.setVelocity(-fr)
            self.rear_left_motor.setVelocity(-rl)
            self.rear_right_motor.setVelocity(rr)
