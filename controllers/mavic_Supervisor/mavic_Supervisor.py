from controller import Supervisor, Emitter, Receiver
import sys
import numpy as np
import math
import struct
import ast
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import box_intersection

class SuperMavic(Supervisor):
    def __init__(self, nameDef):
        Supervisor.__init__(self)
        self.time_step = int(self.getBasicTimeStep())
        self.mavic = self.getFromDef(nameDef)        
        if self.mavic is None:
            print("No Mavic found in the current world file.")
            sys.exit(1)
        self.children_field = self.mavic.getField("children")
        self.body_slot = self.children_field.getMFNode(0)
        self.body_slot_children_field = self.body_slot.getField("children")
        self.pose_of_children_field = self.body_slot_children_field.getMFNode(0)
        self.children_of_pose_field = self.pose_of_children_field.getField("children")
        self.shape_node = self.children_of_pose_field.getMFNode(0)
        self.geometry_field = self.shape_node.getField("geometry")
        self.coord_field = self.geometry_field.getSFNode().getField("coord").getSFNode()
        self.point_field = self.coord_field.getField("point")
        
        self.points = []
        for i in range(20):
            self.points.append(self.point_field.getMFVec3f(i))
        
        self.top_indexes = [4, 5, 6, 7, 16, 17, 18, 19]
        self.bottom_indexes = [0, 1, 2, 3, 8, 9, 11, 13]
        self.front_indexes = [1, 3, 5, 7, 11, 12, 15, 18]
        self.back_indexes = [0, 2, 4, 6, 9, 10, 14, 17]
        self.left_indexes = [2, 3, 6, 7, 13, 14, 15, 19]
        self.right_indexes = [0, 1, 4, 5, 8, 10, 12, 16]
        
        self.top_front_indexes = [point for point in self.top_indexes if point in self.front_indexes]
        self.top_back_indexes = [point for point in self.top_indexes if point in self.back_indexes]
        self.top_left_indexes = [point for point in self.top_indexes if point in self.left_indexes]
        self.top_right_indexes = [point for point in self.top_indexes if point in self.right_indexes]
        self.bottom_front_indexes = [point for point in self.bottom_indexes if point in self.front_indexes]
        self.bottom_back_indexes = [point for point in self.bottom_indexes if point in self.back_indexes]
        self.bottom_left_indexes = [point for point in self.bottom_indexes if point in self.left_indexes]
        self.bottom_right_indexes = [point for point in self.bottom_indexes if point in self.right_indexes]
        self.front_right_indexes = [point for point in self.front_indexes if point in self.right_indexes]
        self.front_left_indexes = [point for point in self.front_indexes if point in self.left_indexes]
        self.back_right_indexes = [point for point in self.back_indexes if point in self.right_indexes]
        self.back_left_indexes = [point for point in self.back_indexes if point in self.left_indexes]

        self.connections= self.geometry_field.getSFNode().getField("coordIndex")
        self.coordIndex = [self.connections.getMFInt32(i) for i in range(self.connections.getCount())]
        self.triangles=[[0, 8, 9], [0, 8, 10], [0, 9, 10], [1, 8, 11], [1, 8, 12], [1, 11, 12],
                        [2, 9, 13], [2, 9, 14], [2, 13, 14], [3, 11, 13], [3, 11, 15], [3, 13, 15],
                        [4, 10, 16], [4, 10, 17], [4, 16, 17], [5, 12, 16], [5, 12, 18], [5, 16, 18],
                        [6, 14, 17], [6, 14, 19], [6, 17, 19], [7, 15, 18], [7, 15, 19], [7, 18, 19],
                        [8, 9, 11], [8, 10, 12], [9, 10, 17], [9, 11, 13], [9, 14, 17], [10, 12, 16],
                        [11, 12, 18], [11, 15, 18], [13, 14, 15], [14, 15, 19], [16, 17, 18], [17, 18, 19]]
        
        self.isNAN = False
        #print("Initial points:", self.points)
        self.mavic.getEmitter = Emitter("emitter")
        self.mavic.getReceiver = Receiver("receiver")
        self.emitter = self.mavic.getEmitter
        self.receiver = self.mavic.getReceiver
        
        self.nameDef = nameDef

    def calculateSpeed(self):
        time_step = int(self.getBasicTimeStep())
        while self.step(time_step) != -1:
            
            # Get the current position of the drone
            position1 = np.array([self.mavic.getPosition()[0], self.mavic.getPosition()[1], self.mavic.getPosition()[2]])

            # Wait for 0.5 second
            self.step(500)

            # Get the new position of the drone
            position2 = np.array([self.mavic.getPosition()[0], self.mavic.getPosition()[1], self.mavic.getPosition()[2]])

            # Calculate the difference in position
            position_difference = position2 - position1

            # Calculate the speed in each dimension
            speed = position_difference / 0.5  # Time difference is 0.5 second

            # Print the speed in each dimension (in meters per second)
            # print("Speed in x direction:", speed[0])
            # print("Speed in y direction:", speed[1])
            # print("Speed in z direction:", speed[2])
            #print(self.nameDef + " Speed ", speed)
            return speed
    
    
    def change_bbox(self):
        speed_vector = self.calculateSpeed()
        
        self.orientation = self.mavic.getOrientation()
        self.x_orientation = [self.orientation[0], self.orientation[3], self.orientation[6]]
        self.y_orientation = [self.orientation[1], self.orientation[4], self.orientation[7]]
        self.z_orientation = [self.orientation[2], self.orientation[5], self.orientation[8]]
        self.position = self.mavic.getPosition()
        # Normalize the speed vector
        speed_vector = speed_vector / np.linalg.norm(speed_vector)
        
        scaled_points = []
        for i in range(20):
            # Scale the points based on the speed in each direction
            point = self.points[i]
            #Cases for the direction of the drone and the sign of the speed vector components 
            #also the rotation of the drone to change the points that are at the side of the drone 
            #that is moving
            #If the drone is looking in the direction of the positive x axis i have the following cases
            #i) If the drone is moving in the positive x direction, the points that are at the front side of the drone should be moved
            #  by the speed vector in the x direction
            #ii) If the drone is moving in the negative x direction, the points that are at the back side of the drone should be moved
            #  by the speed vector in the x direction
            #iii) If the drone is moving in the positive y direction, the points that are at the right side of the drone should be moved
            #  by the speed vector in the y direction
            #iv) If the drone is moving in the negative y direction, the points that are at the left side of the drone should be moved
            #  by the speed vector in the y direction
            #v) If the drone is moving in the positive z direction, the points that are at the top side of the drone should be moved
            #  by the speed vector in the z direction
            #vi) If the drone is moving in the negative z direction, the points that are at the bottom side of the drone should be moved
            #  by the speed vector in the z direction
            #If the drone is looking in the direction of the negative x axis i have the following cases
            #i) If the drone is moving in the positive x direction, the points that are at the back side of the drone should be moved
            #  by the speed vector in the x direction
            #ii) If the drone is moving in the negative x direction, the points that are at the front side of the drone should be moved 
            #  by the speed vector in the x direction
            #iii) If the drone is moving in the positive y direction, the points that are at the left side of the drone should be moved
            #  by the speed vector in the y direction
            #iv) If the drone is moving in the negative y direction, the points that are at the right side of the drone should be moved
            #  by the speed vector in the y direction
            #v) If the drone is moving in the positive z direction, the points that are at the top side of the drone should be moved
            #  by the speed vector in the z direction
            #vi) If the drone is moving in the negative z direction, the points that are at the bottom side of the drone should be moved
            #  by the speed vector in the z direction
            #If the drone is looking in the direction of the positive y axis i have the following cases
            #i) If the drone is moving in the positive x direction, the points that are at the left side of the drone should be moved
            #  by the speed vector in the x direction
            #ii) If the drone is moving in the negative x direction, the points that are at the right side of the drone should be moved
            #  by the speed vector in the x direction
            #iii) If the drone is moving in the positive y direction, the points that are at the front side of the drone should be moved
            #  by the speed vector in the y direction
            #iv) If the drone is moving in the negative y direction, the points that are at the back side of the drone should be moved
            #  by the speed vector in the y direction
            #v) If the drone is moving in the positive z direction, the points that are at the top side of the drone should be moved
            #  by the speed vector in the z direction
            #vi) If the drone is moving in the negative z direction, the points that are at the bottom side of the drone should be moved
            #  by the speed vector in the z direction
            #If the drone is looking in the direction of the negative y axis i have the following cases
            #i) If the drone is moving in the positive x direction, the points that are at the right side of the drone should be moved
            #  by the speed vector in the x direction
            #ii) If the drone is moving in the negative x direction, the points that are at the left side of the drone should be moved
            #  by the speed vector in the x direction
            #iii) If the drone is moving in the positive y direction, the points that are at the back side of the drone should be moved
            #  by the speed vector in the y direction
            #iv) If the drone is moving in the negative y direction, the points that are at the front side of the drone should be moved
            #  by the speed vector in the y direction
            #v) If the drone is moving in the positive z direction, the points that are at the top side of the drone should be moved
            #  by the speed vector in the z direction
            #vi) If the drone is moving in the negative z direction, the points that are at the bottom side of the drone should be moved    
            #  by the speed vector in the z direction
            
            speed_accuracy = 0.15 # speed accuracy to consider the drone is moving in a direction
            scale_factor = 1 # scale factor to move the points based on the speed vector
            #print(self.nameDef + " orientation ", self.orientation)   
            movement = [-1, -1, -1] # front/back (0,1), left/right(0,1), top/bottom(0,1)
            if self.position[2]> 0.1:
                
                if self.y_orientation[0] > - math.sqrt(2)/2 and  self.y_orientation[0] < math.sqrt(2)/2 and self.x_orientation[0] > math.sqrt(2)/2 :
                    print("Drone" + self.nameDef + " is looking in the direction of the positive x axis")
                    if speed_vector[0] > speed_accuracy and speed_vector[1] > speed_accuracy and i in self.front_left_indexes and not changed:
                        new_point = [point[i] + speed_vector[i] * scale_factor for i in range(3)]
                        changed=True
                    elif speed_vector[0] < - speed_accuracy and speed_vector[1] > speed_accuracy and i in self.back_left_indexes and not changed:
                        new_point = [point[i] + speed_vector[i] * scale_factor for i in range(3)]
                        changed=True
                    elif speed_vector[0] > speed_accuracy and speed_vector[1] < - speed_accuracy and i in self.front_right_indexes and not changed:
                        new_point = [point[i] + speed_vector[i] * scale_factor for i in range(3)]
                        changed=True
                    elif speed_vector[0] < - speed_accuracy and speed_vector[1] < - speed_accuracy and i in self.back_right_indexes and not changed:
                        new_point = [point[i] + speed_vector[i] * scale_factor for i in range(3)]
                        changed=True
                    
                    if speed_vector[0] > speed_accuracy and speed_vector[2] > speed_accuracy and i in self.top_front_indexes and not changed:
                        new_point = [point[i] + speed_vector[i] * scale_factor for i in range(3)]
                        changed=True
                    elif speed_vector[0] < - speed_accuracy and speed_vector[2] > speed_accuracy and i in self.top_back_indexes and not changed:
                        new_point = [point[i] + speed_vector[i] * scale_factor for i in range(3)]
                        changed=True
                    elif speed_vector[0] > speed_accuracy and speed_vector[2] < - speed_accuracy and i in self.bottom_front_indexes and not changed:
                        new_point = [point[i] + speed_vector[i] * scale_factor for i in range(3)]
                        changed=True
                    elif speed_vector[0] < - speed_accuracy and speed_vector[2] < - speed_accuracy and i in self.bottom_back_indexes and not changed:
                        new_point = [point[i] + speed_vector[i] * scale_factor for i in range(3)]
                        changed=True
                    
                    if speed_vector[1] > speed_accuracy and speed_vector[2] > speed_accuracy and i in self.top_left_indexes and not changed:
                        new_point = [point[i] + speed_vector[i] * scale_factor for i in range(3)]
                        changed=True
                    elif speed_vector[1] < - speed_accuracy and speed_vector[2] > speed_accuracy and i in self.top_right_indexes and not changed:
                        new_point = [point[i] + speed_vector[i] * scale_factor for i in range(3)]
                        changed=True
                    elif speed_vector[1] > speed_accuracy and speed_vector[2] < - speed_accuracy and i in self.bottom_left_indexes and not changed:
                        new_point = [point[i] + speed_vector[i] * scale_factor for i in range(3)]
                        changed=True
                    elif speed_vector[1] < - speed_accuracy and speed_vector[2] < - speed_accuracy and i in self.bottom_right_indexes and not changed:
                        new_point = [point[i] + speed_vector[i] * scale_factor for i in range(3)]
                        changed=True
                        
                    if speed_vector[0] > speed_accuracy and i in self.front_indexes and not changed:
                        # Moving in positive X direction, adjust points at front side
                        new_point = [point[i] + speed_vector[i] * scale_factor for i in range(3)]
                        changed=True
                    elif speed_vector[0] < - speed_accuracy and i in self.back_indexes and not changed:
                        # Moving in negative X direction, adjust points at back side
                        new_point = [point[i] + speed_vector[i] * scale_factor for i in range(3)]
                        changed=True
        
                    if speed_vector[1] > speed_accuracy and i in self.left_indexes and not changed:
                        # Moving in positive Y direction, adjust points at left side
                        new_point = [point[i] + speed_vector[i] * scale_factor for i in range(3)]
                        changed=True
                    elif speed_vector[1] < - speed_accuracy and i in self.right_indexes and not changed:
                        # Moving in negative Y direction, adjust points at right side
                        new_point = [point[i] + speed_vector[i] * scale_factor for i in range(3)]
                        changed=True
        
                    if speed_vector[2] > speed_accuracy and i in self.top_indexes and not changed:
                        # Moving in positive Z direction, adjust points at top side
                        new_point = [point[i] + speed_vector[i] * scale_factor * (1 if i != 1 else -1) for i in range(3)]
                        changed=True
                    elif speed_vector[2] < - speed_accuracy and i in self.bottom_indexes and not changed:
                        # Moving in negative Z direction, adjust points at bottom side
                        new_point = [point[i] + speed_vector[i] * scale_factor * (1 if i != 1 else -1) for i in range(3)]
                        changed=True
                    
                    
                        
                if self.y_orientation[0] > - math.sqrt(2)/2 and  self.y_orientation[0] < math.sqrt(2)/2 and self.x_orientation[0] < - math.sqrt(2)/2 :
                    print("Drone" + self.nameDef + " is looking in the direction of the negative x axis")
                    if speed_vector[0] > speed_accuracy and speed_vector[1] > speed_accuracy and i in self.back_right_indexes and not changed:
                        new_point = [point[0] - speed_vector[0] * scale_factor, point[1] + speed_vector[1] * scale_factor, point[2]]
                        changed=True
                    elif speed_vector[0] < - speed_accuracy and speed_vector[1] > speed_accuracy and i in self.front_right_indexes and not changed:
                        new_point = [point[0] - speed_vector[0] * scale_factor, point[1] + speed_vector[1] * scale_factor, point[2]]
                        changed=True
                    elif speed_vector[0] > speed_accuracy and speed_vector[1] < - speed_accuracy and i in self.back_left_indexes and not changed:
                        new_point = [point[0] - speed_vector[0] * scale_factor, point[1] - speed_vector[1] * scale_factor, point[2]]
                        changed=True
                    elif speed_vector[0] < - speed_accuracy and speed_vector[1] < - speed_accuracy and i in self.front_left_indexes and not changed:
                        new_point = [point[0] - speed_vector[0] * scale_factor, point[1] - speed_vector[1] * scale_factor, point[2]]
                        changed=True
                    
                    if speed_vector[0] > speed_accuracy and speed_vector[2] > speed_accuracy and i in self.top_back_indexes and not changed:
                        new_point = [point[0] + speed_vector[0] * scale_factor, point[1], point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[0] < - speed_accuracy and speed_vector[2] > speed_accuracy and i in self.top_front_indexes and not changed:
                        new_point = [point[0] + speed_vector[0] * scale_factor, point[1], point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[0] > speed_accuracy and speed_vector[2] < - speed_accuracy and i in self.bottom_back_indexes and not changed:
                        new_point = [point[0] + speed_vector[0] * scale_factor, point[1], point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[0] < - speed_accuracy and speed_vector[2] < - speed_accuracy and i in self.bottom_front_indexes and not changed:
                        new_point = [point[0] + speed_vector[0] * scale_factor, point[1], point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    
                    if speed_vector[1] > speed_accuracy and speed_vector[2] > speed_accuracy and i in self.top_right_indexes and not changed:
                        new_point = [point[0], point[1] + speed_vector[1] * scale_factor, point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[1] < - speed_accuracy and speed_vector[2] > speed_accuracy and i in self.top_left_indexes and not changed:
                        new_point = [point[0], point[1] + speed_vector[1] * scale_factor, point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[1] > speed_accuracy and speed_vector[2] < - speed_accuracy and i in self.bottom_right_indexes and not changed:
                        new_point = [point[0], point[1] + speed_vector[1] * scale_factor, point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[1] < - speed_accuracy and speed_vector[2] < - speed_accuracy and i in self.bottom_left_indexes and not changed:
                        new_point = [point[0], point[1] + speed_vector[1] * scale_factor, point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    
                    if speed_vector[0] > speed_accuracy and i in self.back_indexes and not changed:
                        # Moving in positive X direction, adjust points at back side
                        new_point = [point[0] + speed_vector[0] * scale_factor, point[1], point[2]]
                        changed=True
                    elif speed_vector[0] < -speed_accuracy and i in self.front_indexes and not changed:
                        # Moving in negative X direction, adjust points at front side
                        new_point = [point[0] - speed_vector[0] * scale_factor, point[1], point[2]]
                        changed=True
                    
                    if speed_vector[1] > speed_accuracy and i in self.right_indexes and not changed:
                        # Moving in positive Y direction, adjust points at right side
                        new_point = [point[0], point[1] + speed_vector[1] * scale_factor, point[2]]
                        changed = True
                    elif speed_vector[1] < -speed_accuracy and i in self.left_indexes and not changed:
                        # Moving in negative Y direction, adjust points at left side
                        new_point = [point[0], point[1] - speed_vector[1] * scale_factor, point[2]]
                        changed = True
                    
                    if speed_vector[2] > speed_accuracy and i in self.top_indexes and not changed:
                        # Moving in positive Z direction, adjust points at top side
                        new_point = [point[0], point[1], point[2] + speed_vector[2] * scale_factor]
                        changed = True
                    elif speed_vector[2] < -speed_accuracy and i in self.bottom_indexes and not changed:
                        # Moving in negative Z direction, adjust points at bottom side
                        new_point = [point[0], point[1], point[2] + speed_vector[2] * scale_factor]
                        changed = True


                if self.x_orientation[0] > - math.sqrt(2)/2 and  self.x_orientation[0] < math.sqrt(2)/2 and self.y_orientation[0] < - math.sqrt(2)/2 :
                    print("Drone" + self.nameDef + " is looking in the direction of the positive y axis")
                    if speed_vector[0] > speed_accuracy and speed_vector[1] > speed_accuracy and i in self.front_right_indexes and not changed:
                        new_point = [point[0] + speed_vector[0] * scale_factor, point[1] + speed_vector[1] * scale_factor, point[2]]
                        changed=True
                    elif speed_vector[0] < - speed_accuracy and speed_vector[1] > speed_accuracy and i in self.front_left_indexes and not changed:
                        new_point = [point[0] + speed_vector[0] * scale_factor, point[1] + speed_vector[1] * scale_factor, point[2]]
                        changed=True
                    elif speed_vector[0] > speed_accuracy and speed_vector[1] < - speed_accuracy and i in self.back_right_indexes and not changed:
                        new_point = [point[0] + speed_vector[0] * scale_factor, point[1] + speed_vector[1] * scale_factor, point[2]]
                        changed=True
                    elif speed_vector[0] < - speed_accuracy and speed_vector[1] < - speed_accuracy and i in self.back_left_indexes and not changed:
                        new_point = [point[0] + speed_vector[0] * scale_factor, point[1] + speed_vector[1] * scale_factor, point[2]]
                        changed=True
                    
                    if speed_vector[0] > speed_accuracy and speed_vector[2] > speed_accuracy and i in self.top_right_indexes and not changed:
                        new_point = [point[0] + speed_vector[0] * scale_factor, point[1], point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[0] < - speed_accuracy and speed_vector[2] > speed_accuracy and i in self.top_left_indexes and not changed:
                        new_point = [point[0] + speed_vector[0] * scale_factor, point[1], point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[0] > speed_accuracy and speed_vector[2] < - speed_accuracy and i in self.bottom_right_indexes and not changed:
                        new_point = [point[0] + speed_vector[0] * scale_factor, point[1], point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[0] < - speed_accuracy and speed_vector[2] < - speed_accuracy and i in self.bottom_left_indexes and not changed:
                        new_point = [point[0] + speed_vector[0] * scale_factor, point[1], point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    
                    if speed_vector[1] > speed_accuracy and speed_vector[2] > speed_accuracy and i in self.top_front_indexes and not changed:
                        new_point = [point[0], point[1] + speed_vector[1] * scale_factor, point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[1] < - speed_accuracy and speed_vector[2] > speed_accuracy and i in self.top_back_indexes and not changed:
                        new_point = [point[0], point[1] + speed_vector[1] * scale_factor, point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[1] > speed_accuracy and speed_vector[2] < - speed_accuracy and i in self.bottom_front_indexes and not changed:
                        new_point = [point[0], point[1] + speed_vector[1] * scale_factor, point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[1] < - speed_accuracy and speed_vector[2] < - speed_accuracy and i in self.bottom_back_indexes and not changed:
                        new_point = [point[0], point[1] + speed_vector[1] * scale_factor, point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    
                    if speed_vector[0] > speed_accuracy and i in self.right_indexes and not changed:
                        # Moving in positive X direction, adjust points at right side
                        new_point = [point[0] + speed_vector[0] * scale_factor, point[1], point[2]]
                        changed=True
                    elif speed_vector[0] < - speed_accuracy and i in self.left_indexes and not changed:
                        # Moving in negative X direction, adjust points at left side
                        new_point = [point[0] + speed_vector[0] * scale_factor, point[1], point[2]]
                        changed=True
                    
                    if speed_vector[1] > speed_accuracy and i in self.front_indexes and not changed:
                        # Moving in positive Y direction, adjust points at front side
                        new_point = [point[0], point[1] + speed_vector[1] * scale_factor, point[2]]
                        changed=True
                    elif speed_vector[1] < - speed_accuracy and i in self.back_indexes and not changed:
                        # Moving in negative Y direction, adjust points at back side
                        new_point = [point[0], point[1] + speed_vector[1] * scale_factor, point[2]]
                        changed=True
                    
                    if speed_vector[2] > speed_accuracy and i in self.top_indexes and not changed:
                        # Moving in positive Z direction, adjust points at top side
                        new_point = [point[0], point[1], point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[2] < - speed_accuracy and i in self.bottom_indexes and not changed:
                        # Moving in negative Z direction, adjust points at bottom side
                        new_point = [point[0], point[1], point[2] + speed_vector[2] * scale_factor]
                        changed=True


                if self.x_orientation[0] > - math.sqrt(2)/2 and  self.x_orientation[0] < math.sqrt(2)/2 and self.y_orientation[0] > math.sqrt(2)/2 :
                    print("Drone" + self.nameDef + " is looking in the direction of the negative y axis")
                    if speed_vector[0] > speed_accuracy and speed_vector[1] > speed_accuracy and i in self.back_left_indexes and not changed:
                        new_point = [point[0] + speed_vector[0] * scale_factor, point[1] + speed_vector[1] * scale_factor, point[2]]
                        changed=True
                    elif speed_vector[0] < - speed_accuracy and speed_vector[1] > speed_accuracy and i in self.back_right_indexes and not changed:
                        new_point = [point[0] + speed_vector[0] * scale_factor, point[1] + speed_vector[1] * scale_factor, point[2]]
                        changed=True
                    elif speed_vector[0] > speed_accuracy and speed_vector[1] < - speed_accuracy and i in self.front_left_indexes and not changed:
                        new_point = [point[0] + speed_vector[0] * scale_factor, point[1] + speed_vector[1] * scale_factor, point[2]]
                        changed=True
                    elif speed_vector[0] < - speed_accuracy and speed_vector[1] < - speed_accuracy and i in self.front_right_indexes and not changed:
                        new_point = [point[0] + speed_vector[0] * scale_factor, point[1] + speed_vector[1] * scale_factor, point[2]]
                        changed=True
                    
                    if speed_vector[0] > speed_accuracy and speed_vector[2] > speed_accuracy and i in self.top_left_indexes and not changed:
                        new_point = [point[0] + speed_vector[0] * scale_factor, point[1], point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[0] < - speed_accuracy and speed_vector[2] > speed_accuracy and i in self.top_right_indexes and not changed:
                        new_point = [point[0] + speed_vector[0] * scale_factor, point[1], point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[0] > speed_accuracy and speed_vector[2] < - speed_accuracy and i in self.bottom_left_indexes and not changed:
                        new_point = [point[0] + speed_vector[0] * scale_factor, point[1], point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[0] < - speed_accuracy and speed_vector[2] < - speed_accuracy and i in self.bottom_right_indexes and not changed:
                        new_point = [point[0] + speed_vector[0] * scale_factor, point[1], point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    
                    if speed_vector[1] > speed_accuracy and speed_vector[2] > speed_accuracy and i in self.top_back_indexes and not changed:
                        new_point = [point[0], point[1] + speed_vector[1] * scale_factor, point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[1] < - speed_accuracy and speed_vector[2] > speed_accuracy and i in self.top_front_indexes and not changed:
                        new_point = [point[0], point[1] + speed_vector[1] * scale_factor, point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[1] > speed_accuracy and speed_vector[2] < - speed_accuracy and i in self.bottom_back_indexes and not changed:
                        new_point = [point[0], point[1] + speed_vector[1] * scale_factor, point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[1] < - speed_accuracy and speed_vector[2] < - speed_accuracy and i in self.bottom_front_indexes and not changed:
                        new_point = [point[0], point[1] + speed_vector[1] * scale_factor, point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    
                    if speed_vector[0] > speed_accuracy and i in self.left_indexes and not changed:
                        # Moving in positive X direction, adjust points at left side
                        new_point = [point[0] + speed_vector[0] * scale_factor, point[1], point[2]]
                        changed=True
                    elif speed_vector[0] < -speed_accuracy and i in self.right_indexes and not changed:
                        # Moving in negative X direction, adjust points at right side
                        new_point = [point[0] + speed_vector[0] * scale_factor, point[1], point[2]]
                        changed=True
                    
                    if speed_vector[1] > speed_accuracy and i in self.back_indexes and not changed:
                        # Moving in positive Y direction, adjust points at back side
                        new_point = [point[0], point[1] + speed_vector[1] * scale_factor, point[2]]
                        changed = True
                    elif speed_vector[1] < -speed_accuracy and i in self.front_indexes and not changed:
                        # Moving in negative Y direction, adjust points at front side
                        new_point = [point[0], point[1] + speed_vector[1] * scale_factor, point[2]]
                        changed = True
                    
                    if speed_vector[2] > speed_accuracy and i in self.top_indexes and not changed:
                        # Moving in positive Z direction, adjust points at top side
                        new_point = [point[0], point[1], point[2] + speed_vector[2] * scale_factor]
                        changed = True
                    elif speed_vector[2] < -speed_accuracy and i in self.bottom_indexes and not changed:
                        # Moving in negative Z direction, adjust points at bottom side
                        new_point = [point[0], point[1], point[2] + speed_vector[2] * scale_factor]
                        changed = True
                        
            if not changed:
                new_point = self.points[i][:]  

            #new_speed_point = [new_point[j] + (abs(speed_vector[j]) if j!=2 else speed_vector[j]) * scale_factor for j in range(3)]

            # Append the scaled point to the list
            scaled_points.append(new_point)
                

        # Update the bounding box using the scaled points
        self.updateBoundingBox(scaled_points)
    
    def updateBoundingBox(self, points):
        # Update the bounding box using the new points
        for i in range(20):
            self.point_field.setMFVec3f(i, points[i])
            #print("Point ", i, ":", points[i])
            #print("self.point", i, ":", self.points[i])

    def findPointsFromMessage(self, message):
        
        # Extracting the string representation of the box vertices
        start_index = message.find("[")
        end_index = message.rfind("]") + 1  # Add 1 to include the closing bracket
        box_vertices_str = message[start_index:end_index]
        vertices_list = box_vertices_str.split(",")
        vertices_list = [coord.strip("()") for coord in vertices_list]
        vertices_list = [coord.replace("(", "") for coord in vertices_list]
        vertices_list = [coord.replace(")", "") for coord in vertices_list]
        vertices_list = [coord.replace("[", "") for coord in vertices_list]
        vertices_list = [coord.replace("]", "") for coord in vertices_list]
        vertices_list = [coord.replace(" ", "") for coord in vertices_list]
        vertices_list = [coord.split(",") for coord in vertices_list]
        vertices_list = [[float(coord) for coord in vertex] for vertex in vertices_list]
        vertices_list = [vertex for sublist in vertices_list for vertex in sublist]
        # Combine the vertices into triples
        points = [tuple(vertices_list[i:i+3]) for i in range(0, len(vertices_list), 3)]

        #print(points)
        return points
            
    def findCollision(self, box1, box2):
        box1 = box_intersection.Box(box1)
        box2 = box_intersection.Box(box2)
        return box_intersection.boxes_intersect(box1, box2)
        
    def run(self):
        while self.step(self.time_step) != -1:
            
            self.change_bbox()
            # message="bbox of "+self.nameDef+" "
            # self.emitter.send(message)
            # # Example: Receive a message on the receiver
            # if self.receiver.getQueueLength() > 0:
            #     received_message = self.receiver.getString()
            #     #print("Received message to "+self.nameDef+":" , received_message)
            #     self.receiver.nextPacket()  # Move to the next received packet

            #     box1=self.calculateVerticesOfBox()                

                
            #     box2 = self.findPointsFromMessage(received_message)
                
            #     collision = self.findCollision(box1, box2)
            #     # if collision:
            #     #     print("Collision detected from " + self.nameDef + " with the other drone.")
            #     #     # Handle collision logic here
            #     # else:
            #     #     print("No collision detected from " + self.nameDef + " with the other drone.")

            
            self.simulationResetPhysics()
            
            if self.isNAN:
               break
            if (self.calculateSpeed()[2]<0) and self.mavic.getPosition()[2] < 0.1:
                print("Landed")
                break
        print("Exiting...")
        sys.exit(0)