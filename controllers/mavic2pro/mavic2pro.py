from controller import Robot, Camera
import cv2
import numpy as np

# Create the robot instance
robot = Robot()
camera = robot.getDevice("camera")
camera.enable(int(robot.getBasicTimeStep()))  # Convert to integer

# ArUco parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# Target ArUco ID to land on
target_id = 0

# Drone control parameters
MAX_DESCENT_VELOCITY = -0.1  # Adjust this value based on your simulator and requirements

# Get propeller motors and set them to velocity mode
front_left_motor = robot.getDevice("front left propeller")
front_right_motor = robot.getDevice("front right propeller")
rear_left_motor = robot.getDevice("rear left propeller")
rear_right_motor = robot.getDevice("rear right propeller")
motors = [front_left_motor, front_right_motor, rear_left_motor, rear_right_motor]
for motor in motors:
    motor.setPosition(float('inf'))
    motor.setVelocity(0.0)

while robot.step(int(robot.getBasicTimeStep())) != -1:
    # Capture image from the camera
    image = camera.getImage()

    # Check if the image is not None
    if image is not None:
        # Convert the image to grayscale
        image_array = np.frombuffer(image, dtype=np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # Search for the target ArUco marker
        if ids is not None and target_id in ids:
            target_index = list(ids).index(target_id)
            target_corners = corners[target_index][0]

            # Get the center of the detected marker
            cx, cy = np.mean(target_corners, axis=0)

            # Implement landing logic based on marker position
            # For simplicity, let's assume landing when the marker is centered
            if abs(cx - camera.getWidth() / 2) < 50 and abs(cy - camera.getHeight() / 2) < 50:
                # Land the drone
                for motor in motors:
                    motor.setVelocity(MAX_DESCENT_VELOCITY)

        # Display the image with detected markers (for visualization purposes)
        img_with_markers = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if corners:
            img_with_markers = cv2.aruco.drawDetectedMarkers(img_with_markers, corners, ids)
        cv2.imshow("Webots Camera", img_with_markers)
        cv2.waitKey(1)
