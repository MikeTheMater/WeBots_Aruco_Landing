from controller import Robot, Supervisor
import sys
import numpy as np
import cv2
import math
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import box_intersection

def clamp(value, value_min, value_max):
    return min(max(value, value_min), value_max)

class Mavic(Robot):
    # Constants, empirically found.
    K_VERTICAL_THRUST = 68.5  # with this thrust, the drone lifts.
    K_VERTICAL_OFFSET = 0.6   # Vertical offset where the robot actually targets to stabilize itself.
    K_VERTICAL_P = 3.0        # P constant of the vertical PID.
    K_ROLL_P = 50.0           # P constant of the roll PID.
    K_PITCH_P = 30.0          # P constant of the pitch PID.
    MAX_YAW_DISTURBANCE = 0.4
    MAX_PITCH_DISTURBANCE = -1
    target_precision = 0.05    # Precision between the target position and the robot position in meters

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
        self.marker_position = [0, 0]
        self.target_index = 0
        self.landed = False

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
        self.target_position[0:2] = [x, y]

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
        pitch_disturbance = clamp(
            self.MAX_PITCH_DISTURBANCE * distance_to_target / 10, 
            self.MAX_PITCH_DISTURBANCE, 0
        )

        return yaw_disturbance, pitch_disturbance
        
    def calculate_translation_rel_to_world(self, translation):
        return [translation[0] + self.gps.getValues()[0], translation[1] + self.gps.getValues()[1], translation[2] + self.gps.getValues()[2]]
    
    def calculate_speed(self):
        time_step = int(self.getBasicTimeStep())
        while self.step(time_step) != -1:
            
            # Get the current position of the drone
            position1 = np.array([self.gps.getValues()[0], self.gps.getValues()[1], self.gps.getValues()[2]])

            # Wait for 1 second
            self.step(500)

            # Get the new position of the drone
            position2 = np.array([self.gps.getValues()[0], self.gps.getValues()[1], self.gps.getValues()[2]])

            # Calculate the difference in position
            position_difference = position2 - position1

            # Calculate the speed in each dimension
            speed = position_difference / 0.5  # Time difference is 1 second

            # Print the speed in each dimension (in meters per second)
            #print("Speed in x direction:", speed[0])
            #print("Speed in y direction:", speed[1])
            #print("Speed in z direction:", speed[2])
            return speed
    
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
            self.marker_position = [x_world, y_world]
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
        descent_rate = 0.01  # Meters per time step, adjust as needed
        touchdown_altitude = 0.2  # Altitude at which to cut off motors, adjust as needed

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

    def move_to_target(self, waypoints, verbose_movement=False, verbose_target=False):
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
            if verbose_target:
                print("First target: ", self.target_position[0:2])

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
        angle_left = self.target_position[2] - self.current_pose[5]
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
    
    def move_right_by_motor_control(self, speed_difference, target_altitude=5):
        # Constants (adjust as needed)
        k_roll_p = self.K_ROLL_P
        k_pitch_p = self.K_PITCH_P
        k_vertical_p = self.K_VERTICAL_P
        k_vertical_thrust = self.K_VERTICAL_THRUST
        k_vertical_offset = self.K_VERTICAL_OFFSET

        # Base motor speed (this can be adjusted)
        base_speed = 1.0

        # Get current sensor values
        roll = self.imu.getRollPitchYaw()[0]  # Assuming roll is the first value
        pitch = self.imu.getRollPitchYaw()[1]  # Assuming pitch is the second value
        altitude = self.gps.getValues()[2]
        roll_velocity = self.gyro.getValues()[0]
        pitch_velocity = self.gyro.getValues()[1]
        pitch_disturbance = 0
        yaw_disturbance = 0
        roll_disturbance = -1
        # Compute roll, pitch, yaw, and vertical inputs
        roll_input = k_roll_p * clamp(roll, -1.0, 1.0) + roll_velocity + roll_disturbance
        pitch_input = k_pitch_p * clamp(pitch, -1.0, 1.0) + pitch_velocity + pitch_disturbance
        yaw_input = speed_difference  # Use speed_difference to create the right turn

        clamped_difference_altitude = clamp(target_altitude - altitude + k_vertical_offset, -1.0, 1.0)
        vertical_input = k_vertical_p * pow(clamped_difference_altitude, 3.0)

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


        
    def run(self):
        t1 = self.getTime()

        roll_disturbance = 0
        pitch_disturbance = 0
        yaw_disturbance = 0

        # Specify the patrol coordinates
        #waypoints = [[-30, 20], [-60, 20], [-60, 10], [-30, 5]]
        #waypoints = [[-6.36,3.2]]
        # target altitude of the robot in meters
        self.target_altitude = 5

        detected_marker = False
        while self.step(self.time_step) != -1:
            
            #self.com_between_drones()
            # Read sensors and set position.
            roll, pitch, yaw = self.imu.getRollPitchYaw()
            x_pos, y_pos, altitude = self.gps.getValues()
            roll_acceleration, pitch_acceleration, _ = self.gyro.getValues()
            self.set_position([x_pos, y_pos, altitude, roll, pitch, yaw])

            collistion_Status = self.getCustomData()
            if collistion_Status != "0":   
                # If collision detected
                print("Collision detected from controller")
                
                temp_collistion_Status = [float(i) for i in collistion_Status.split(" ")]
                
                self.move_right_by_motor_control(0.1)
                self.set_position([temp_collistion_Status[0:2]])
                continue
            else:
                if not self.marker_detected:
                    if altitude > self.target_altitude - 1:
                        current_time = self.getTime()
                        self.update_sinusoidal_waypoints(current_time)
                        yaw_disturbance, pitch_disturbance = self.compute_movement()
                    else:
                        yaw_disturbance = 0
                        pitch_disturbance = 0
                else:
                    # Move towards the marker position and land
                    self.target_position[0:2] = self.marker_position
                    yaw_disturbance, pitch_disturbance = self.move_to_target([self.marker_position])
                    if abs(self.current_pose[0] - self.marker_position[0]) < 0.5 and abs(self.current_pose[1] - self.marker_position[1]) < 0.5:
                        print("Landing on the marker.")
                        #print("Marker position: ", self.marker_position)
                        #print("Current position: ", self.current_pose[0:2])
                        self.land(altitude)
                        break

                # Detect marker and switch to marker mode if found.
                if not self.marker_detected and self.detect_aruco_marker():
                    print("Marker detected, switching to marker mode.")
                    print("Marker position: ", self.marker_position)
                    self.marker_detected = True

                # Movement logic using 'yaw_disturbance', 'pitch_disturbance', etc.
                roll_input = self.K_ROLL_P * clamp(roll, -1, 1) + roll_acceleration + yaw_disturbance
                pitch_input = self.K_PITCH_P * clamp(pitch, -1, 1) + pitch_acceleration + pitch_disturbance
                yaw_input = yaw_disturbance
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
                    #break
                self.front_left_motor.setVelocity(front_left_motor_input)
                self.front_right_motor.setVelocity(-front_right_motor_input)
                self.rear_left_motor.setVelocity(-rear_left_motor_input)
                self.rear_right_motor.setVelocity(rear_right_motor_input)
            
# Main execution
#robot = Mavic()
#robot.set_id(0)
#robot.run()
