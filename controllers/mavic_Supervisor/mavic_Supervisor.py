from controller import Supervisor, Emitter, Receiver
import sys
import numpy as np
import math
import struct
import ast
import os
import json
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import box_intersection
import Trying_the_normal

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
            #The points of the Bounding box of the drone
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
        #Triangles using the index of the point in the list of points of the box 
        self.triangles= Trying_the_normal.find_triangles(self.points, self.coordIndex)
        #print ("Triangles", self.triangles)
        # Store data from other drones
        self.other_drones_data = []
        
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

            speed=np.array([speed[0], speed[1], speed[2]])

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
        
        center=np.mean(self.points, axis=0)

        # Calculate the normals of the points
        normals = Trying_the_normal.calculate_point_normals(self.points, self.triangles, center)
        threshold=0.6
        points_in_direction, points_opposite_direction = Trying_the_normal.classify_points_by_normal(normals, speed_vector, threshold)


        self.scaled_points = []
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
            changed = False
            
            if self.position[2]> 0.1:
                #Fixed the positive and negative x axises, positive and negative y axis
                if self.y_orientation[0] > - math.sqrt(2)/2 and  self.y_orientation[0] < math.sqrt(2)/2 and self.x_orientation[0] > math.sqrt(2)/2 :
                    #print("Drone" + self.nameDef + " is looking in the direction of the positive x axis")
                    if speed_vector[0] > speed_accuracy and speed_vector[1] > speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] + speed_vector[0] * scale_factor, point[1] + speed_vector[1] * scale_factor, point[2]]
                        changed=True
                        
                    elif speed_vector[0] < - speed_accuracy and speed_vector[1] > speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] + speed_vector[0] * scale_factor, point[1] + speed_vector[1] * scale_factor, point[2]]
                        changed=True
                    elif speed_vector[0] > speed_accuracy and speed_vector[1] < - speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] + speed_vector[0] * scale_factor, point[1] + speed_vector[1] * scale_factor, point[2]]
                        changed=True
                    elif speed_vector[0] < - speed_accuracy and speed_vector[1] < - speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] + speed_vector[0] * scale_factor, point[1] + speed_vector[1] * scale_factor, point[2]]
                        changed=True
                    
                    if speed_vector[0] > speed_accuracy and speed_vector[2] > speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] + speed_vector[0] * scale_factor, point[1], point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[0] < - speed_accuracy and speed_vector[2] > speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] + speed_vector[0] * scale_factor, point[1], point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[0] > speed_accuracy and speed_vector[2] < - speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] + speed_vector[0] * scale_factor, point[1], point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[0] < - speed_accuracy and speed_vector[2] < - speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] + speed_vector[0] * scale_factor, point[1], point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    
                    if speed_vector[1] > speed_accuracy and speed_vector[2] > speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0], point[1] + speed_vector[1] * scale_factor, point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[1] < - speed_accuracy and speed_vector[2] > speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0], point[1] + speed_vector[1] * scale_factor, point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[1] > speed_accuracy and speed_vector[2] < - speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0], point[1] + speed_vector[1] * scale_factor, point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[1] < - speed_accuracy and speed_vector[2] < - speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0], point[1] + speed_vector[1] * scale_factor, point[2] + speed_vector[2] * scale_factor]
                        changed=True
                        
                    if speed_vector[0] > speed_accuracy and i in points_in_direction and not changed:
                        # Moving in positive X direction, adjust points at front side
                        new_point = [point[0] + speed_vector[0] * scale_factor, point[1], point[2]]
                        changed=True
                    
                    elif speed_vector[0] < - speed_accuracy and i in points_in_direction and not changed:
                        # Moving in negative X direction, adjust points at back side
                        new_point = [point[0] + speed_vector[0] * scale_factor, point[1], point[2]]
                        changed=True
        
                    if speed_vector[1] > speed_accuracy and i in points_in_direction and not changed:
                        # Moving in positive Y direction, adjust points at left side
                        new_point = [point[0] , point[1] + speed_vector[1] * scale_factor, point[2]]
                        changed=True
                    elif speed_vector[1] < - speed_accuracy and i in points_in_direction and not changed:
                        # Moving in negative Y direction, adjust points at right side
                        new_point = [point[0] , point[1] + speed_vector[1] * scale_factor, point[2]]
                        changed=True
        
                    if speed_vector[2] > speed_accuracy and i in points_in_direction and not changed:
                        # Moving in positive Z direction, adjust points at top side
                        new_point = [point[0], point[1], point[2] + speed_vector[2] * scale_factor]
                        changed=True
                        print("Moving forward on the positive z axis")
                    elif speed_vector[2] < - speed_accuracy and i in points_in_direction and not changed:
                        # Moving in negative Z direction, adjust points at bottom side
                        new_point = [point[0], point[1], point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    
                    
                        
                if self.y_orientation[0] > - math.sqrt(2)/2 and  self.y_orientation[0] < math.sqrt(2)/2 and self.x_orientation[0] < - math.sqrt(2)/2 :
                    #print("Drone" + self.nameDef + " is looking in the direction of the negative x axis")
                    if speed_vector[0] > speed_accuracy and speed_vector[1] > speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] - speed_vector[0] * scale_factor, point[1] + speed_vector[1] * scale_factor, point[2]]
                        changed=True
                    elif speed_vector[0] < - speed_accuracy and speed_vector[1] > speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] - speed_vector[0] * scale_factor, point[1] - speed_vector[1] * scale_factor, point[2]]
                        changed=True
                    elif speed_vector[0] > speed_accuracy and speed_vector[1] < - speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] - speed_vector[0] * scale_factor, point[1] - speed_vector[1] * scale_factor, point[2]]
                        changed=True
                    elif speed_vector[0] < - speed_accuracy and speed_vector[1] < - speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] - speed_vector[0] * scale_factor, point[1] - speed_vector[1] * scale_factor, point[2]]
                        changed=True
                    
                    if speed_vector[0] > speed_accuracy and speed_vector[2] > speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] - speed_vector[0] * scale_factor, point[1], point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[0] < - speed_accuracy and speed_vector[2] > speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] - speed_vector[0] * scale_factor, point[1], point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[0] > speed_accuracy and speed_vector[2] < - speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] - speed_vector[0] * scale_factor, point[1], point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[0] < - speed_accuracy and speed_vector[2] < - speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] - speed_vector[0] * scale_factor, point[1], point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    
                    if speed_vector[1] > speed_accuracy and speed_vector[2] > speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0], point[1] - speed_vector[1] * scale_factor, point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[1] < - speed_accuracy and speed_vector[2] > speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0], point[1] - speed_vector[1] * scale_factor, point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[1] > speed_accuracy and speed_vector[2] < - speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0], point[1] - speed_vector[1] * scale_factor, point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[1] < - speed_accuracy and speed_vector[2] < - speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0], point[1] - speed_vector[1] * scale_factor, point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    
                    if speed_vector[0] > speed_accuracy and i in points_in_direction and not changed:
                        # Moving in positive X direction, adjust points at back side
                        new_point = [point[0] - speed_vector[0] * scale_factor, point[1], point[2]]
                        changed=True
                    elif speed_vector[0] < -speed_accuracy and i in points_in_direction and not changed:
                        # Moving in negative X direction, adjust points at front side
                        new_point = [point[0] - speed_vector[0] * scale_factor, point[1], point[2]]
                        changed=True
                    
                    if speed_vector[1] > speed_accuracy and i in points_in_direction and not changed:
                        # Moving in positive Y direction, adjust points at right side
                        new_point = [point[0], point[1] - speed_vector[1] * scale_factor, point[2]]
                        changed = True
                    elif speed_vector[1] < -speed_accuracy and i in points_in_direction and not changed:
                        # Moving in negative Y direction, adjust points at left side
                        new_point = [point[0], point[1] - speed_vector[1] * scale_factor, point[2]]
                        changed = True
                    
                    if speed_vector[2] > speed_accuracy and i in points_in_direction and not changed:
                        # Moving in positive Z direction, adjust points at top side
                        new_point = [point[0], point[1], point[2] + speed_vector[2] * scale_factor]
                        changed = True
                    elif speed_vector[2] < -speed_accuracy and i in points_in_direction and not changed:
                        # Moving in negative Z direction, adjust points at bottom side
                        new_point = [point[0], point[1], point[2] + speed_vector[2] * scale_factor]
                        changed = True


                if self.x_orientation[0] > - math.sqrt(2)/2 and  self.x_orientation[0] < math.sqrt(2)/2 and self.y_orientation[0] < - math.sqrt(2)/2 :
                    #print("Drone" + self.nameDef + " is looking in the direction of the positive y axis")
                    if speed_vector[0] > speed_accuracy and speed_vector[1] > speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] + speed_vector[1] * scale_factor, point[1] - speed_vector[0] * scale_factor, point[2]]
                        changed=True                       
                    elif speed_vector[0] < - speed_accuracy and speed_vector[1] > speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] + speed_vector[1] * scale_factor, point[1] - speed_vector[0] * scale_factor, point[2]]
                        changed=True
                    elif speed_vector[0] > speed_accuracy and speed_vector[1] < - speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] + speed_vector[1] * scale_factor, point[1] - speed_vector[0] * scale_factor, point[2]]
                        changed=True
                    elif speed_vector[0] < - speed_accuracy and speed_vector[1] < - speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] + speed_vector[1] * scale_factor, point[1] - speed_vector[0] * scale_factor, point[2]]
                        changed=True
                    
                    if speed_vector[0] > speed_accuracy and speed_vector[2] > speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] , point[1] - speed_vector[0] * scale_factor, point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[0] < - speed_accuracy and speed_vector[2] > speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] , point[1] - speed_vector[0] * scale_factor, point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[0] > speed_accuracy and speed_vector[2] < - speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] , point[1] - speed_vector[0] * scale_factor, point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[0] < - speed_accuracy and speed_vector[2] < - speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] , point[1] - speed_vector[0] * scale_factor, point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    
                    if speed_vector[1] > speed_accuracy and speed_vector[2] > speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] + speed_vector[1] * scale_factor, point[1], point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[1] < - speed_accuracy and speed_vector[2] > speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] + speed_vector[1] * scale_factor, point[1], point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[1] > speed_accuracy and speed_vector[2] < - speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] + speed_vector[1] * scale_factor, point[1], point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[1] < - speed_accuracy and speed_vector[2] < - speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] + speed_vector[1] * scale_factor, point[1], point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    
                    if speed_vector[0] > speed_accuracy and i in points_in_direction and not changed:
                        # Moving in positive X direction, adjust points at right side
                        new_point = [point[0], point[1] - speed_vector[0] * scale_factor, point[2]]
                        changed=True
                    elif speed_vector[0] < - speed_accuracy and i in points_in_direction and not changed:
                        # Moving in negative X direction, adjust points at left side
                        new_point = [point[0], point[1] - speed_vector[0] * scale_factor, point[2]]
                        changed=True
                    
                    if speed_vector[1] > speed_accuracy and i in points_in_direction and not changed:
                        # Moving in positive Y direction, adjust points at front side
                        new_point = [point[0] + speed_vector[1] * scale_factor, point[1], point[2]]
                        changed=True
                    elif speed_vector[1] < - speed_accuracy and i in points_in_direction and not changed:
                        # Moving in negative Y direction, adjust points at back side
                        new_point = [point[0] + speed_vector[1] * scale_factor, point[1], point[2]]
                        changed=True
                    
                    if speed_vector[2] > speed_accuracy and i in points_in_direction and not changed:
                        # Moving in positive Z direction, adjust points at top side
                        new_point = [point[0], point[1], point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[2] < - speed_accuracy and i in points_in_direction and not changed:
                        # Moving in negative Z direction, adjust points at bottom side
                        new_point = [point[0], point[1], point[2] + speed_vector[2] * scale_factor]
                        changed=True


                if self.x_orientation[0] > - math.sqrt(2)/2 and  self.x_orientation[0] < math.sqrt(2)/2 and self.y_orientation[0] > math.sqrt(2)/2 :
                    #print("Drone" + self.nameDef + " is looking in the direction of the negative y axis")
                    if speed_vector[0] > speed_accuracy and speed_vector[1] > speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] - speed_vector[1] * scale_factor, point[1] + speed_vector[0] * scale_factor, point[2]]
                        changed=True
                    elif speed_vector[0] < - speed_accuracy and speed_vector[1] > speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] - speed_vector[1] * scale_factor, point[1] + speed_vector[0] * scale_factor, point[2]]
                        changed=True
                    elif speed_vector[0] > speed_accuracy and speed_vector[1] < - speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] - speed_vector[1] * scale_factor, point[1] + speed_vector[0] * scale_factor, point[2]]
                        changed=True
                    elif speed_vector[0] < - speed_accuracy and speed_vector[1] < - speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] - speed_vector[1] * scale_factor, point[1] + speed_vector[0] * scale_factor, point[2]]
                        changed=True
                    
                    if speed_vector[0] > speed_accuracy and speed_vector[2] > speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] , point[1] + speed_vector[0] * scale_factor, point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[0] < - speed_accuracy and speed_vector[2] > speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] , point[1] + speed_vector[0] * scale_factor, point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[0] > speed_accuracy and speed_vector[2] < - speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] , point[1] + speed_vector[0] * scale_factor, point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[0] < - speed_accuracy and speed_vector[2] < - speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] , point[1] + speed_vector[0] * scale_factor, point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    
                    if speed_vector[1] > speed_accuracy and speed_vector[2] > speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] - speed_vector[1] * scale_factor, point[1], point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[1] < - speed_accuracy and speed_vector[2] > speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] - speed_vector[1] * scale_factor, point[1], point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[1] > speed_accuracy and speed_vector[2] < - speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] - speed_vector[1] * scale_factor, point[1], point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    elif speed_vector[1] < - speed_accuracy and speed_vector[2] < - speed_accuracy and i in points_in_direction and not changed:
                        new_point = [point[0] - speed_vector[1] * scale_factor, point[1], point[2] + speed_vector[2] * scale_factor]
                        changed=True
                    
                    if speed_vector[0] > speed_accuracy and i in points_in_direction and not changed:
                        # Moving in positive X direction, adjust points at left side
                        new_point = [point[0], point[1] + speed_vector[0] * scale_factor, point[2]]
                        changed=True
                    elif speed_vector[0] < -speed_accuracy and i in points_in_direction and not changed:
                        # Moving in negative X direction, adjust points at right side
                        new_point = [point[0], point[1] + speed_vector[0] * scale_factor, point[2]]
                        changed=True
                    
                    if speed_vector[1] > speed_accuracy and i in points_in_direction and not changed:
                        # Moving in positive Y direction, adjust points at back side
                        new_point = [point[0] - speed_vector[1] * scale_factor, point[1], point[2]]
                        changed = True
                    elif speed_vector[1] < -speed_accuracy and i in points_in_direction and not changed:
                        # Moving in negative Y direction, adjust points at front side
                        new_point = [point[0] - speed_vector[1] * scale_factor, point[1], point[2]]
                        changed = True
                    
                    if speed_vector[2] > speed_accuracy and i in points_in_direction and not changed:
                        # Moving in positive Z direction, adjust points at top side
                        new_point = [point[0], point[1], point[2] + speed_vector[2] * scale_factor]
                        changed = True
                    elif speed_vector[2] < -speed_accuracy and i in points_in_direction and not changed:
                        # Moving in negative Z direction, adjust points at bottom side
                        new_point = [point[0], point[1], point[2] + speed_vector[2] * scale_factor]
                        changed = True

                        
            if not changed:
                new_point = self.points[i][:]  

            #new_speed_point = [new_point[j] + (abs(speed_vector[j]) if j!=2 else speed_vector[j]) * scale_factor for j in range(3)]

            # Append the scaled point to the list
            self.scaled_points.append(new_point)
                

        # Update the bounding box using the scaled points
        self.updateBoundingBox(self.scaled_points)
    
    def updateBoundingBox(self, points):
        # Update the bounding box using the new points
        for i in range(20):
            self.point_field.setMFVec3f(i, points[i])
            #print("Point ", i, ":", points[i])
            #print("self.point", i, ":", self.points[i])

    def send_data(self):
        position = self.mavic.getPosition()
        triangles = self.get_triangles()  # Assume this function returns the triangles data
        points_list = [vertex.tolist() for triangle in triangles for vertex in triangle]  # Convert to list of lists
        message = json.dumps({"name": self.nameDef, "position": position, "points": points_list})  # Serialize to JSON
        self.emitter.send(message.encode())  # Send as byte data

    def receive_data(self):
        self.other_drones_data.clear()
        while self.receiver.getQueueLength() > 0:
            received_message = self.receiver.getString()  # Get the message as a string
            data = json.loads(received_message)  # Deserialize JSON
            other_drone_name = data["name"]
            box2 = self.findPointsFromMessage(data["points"])  # Extract points
            self.other_drones_data.append((other_drone_name, box2))
            self.receiver.nextPacket()

            # Print points for both drones
            #print(f"Drone {self.nameDef} has these points: {self.points}")
            #print(f"Drone {other_drone_name} has these points: {box2}")

    def check_collisions(self):
        box1 = self.get_triangles()
        for other_drone_name, other_triangles in self.other_drones_data:
            collision = self.findCollision(box1, other_triangles)
            if collision:
                print(f"Possible collision detected between {self.nameDef} and {other_drone_name}.")
            else:
                print(f"No collision detected between {self.nameDef} and {other_drone_name}.")

    def findPointsFromMessage(self, points_list):
        points = [tuple(points_list[i:i+3]) for i in range(0, len(points_list), 3)]
        return points

    def get_triangles(self):
        # Get the current position of the drone
        position = self.mavic.getPosition()
        
        # Convert self.triangles indices to actual coordinates from self.points
        triangles_coords = []
        for triangle in self.triangles:
            triangle_coords = np.array([self.scaled_points[idx] for idx in triangle])
            global_triangle_coords = triangle_coords + position  # Add position to each point
            triangles_coords.append(global_triangle_coords)
        return triangles_coords
    
        """It doesn't detect the collision correctly
        it doesn't detect the collision of the bbox of the drones but of the drones themselves"""
    def findCollision(self, box1, box2, tolerance=1e-6):
        for triangle1 in box1:
            for triangle2 in box2:
                if box_intersection.triangles_intersect(np.array(triangle1).reshape(3, 3), np.array(triangle2).reshape(3, 3), tolerance=tolerance):
                    print("Possible collision detected on triangles:" , triangle1, triangle2)
                    return True
        return False
        
    def run(self):
        while self.step(self.time_step) != -1:
            
            self.change_bbox()
            self.send_data()
            self.receive_data()
            self.check_collisions()
            self.simulationResetPhysics()
            
            if self.isNAN:
               break
            if (self.calculateSpeed()[2]<0) and self.mavic.getPosition()[2] < 0.1:
                print("Landed")
                break
        print("Exiting...")
        sys.exit(0)