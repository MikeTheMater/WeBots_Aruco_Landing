from controller import Supervisor, Emitter, Receiver
import sys
import numpy as np
import math
import struct
import ast
import os
sys.path.append(os.path.abspath(r"C:\Users\MikeTheMater\Desktop\Landing_Site_Detection"))
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
        
        self.top_indexes = [4, 5, 6, 7, 16, 18, 19, 20]
        self.bottom_indexes = [0, 1, 2, 3, 8, 9, 11, 13]
        self.front_indexes = [2, 3, 6, 7, 13, 14, 15, 19]
        self.back_indexes = [0, 1, 4, 5, 8, 10, 12, 16]
        self.left_indexes = [0, 2, 4, 6, 8, 9, 10, 14, 17]
        self.right_indexes = [1, 3, 5, 7, 9, 11, 12, 13, 18]
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
            #print("Speed in x direction:", speed[0])
            #print("Speed in y direction:", speed[1])
            #print("Speed in z direction:", speed[2])
            return speed
    
    
    def change_bbox(self):
        speed_vector = self.calculateSpeed()

        # Define scaling factors for each direction
        scale_factor = 1  # Adjust as needed
        
        self.rotation_field = self.mavic.getField("rotation")
        self.rotation = self.rotation_field.getSFRotation()
        
        # Normalize the speed vector
        speed_vector = speed_vector / np.linalg.norm(speed_vector)
        
        scaled_points = []
        for i in range(20):
            # Scale the points based on the speed in each direction
            point = self.points[i]
            #Cases for the direction of the drone and the sign of the speed vector components 
            #also the rotation of the drone to change the points that are at the side of the drone 
            #that is moving
            if speed_vector[0] > 0 and self.rotation[3] - 1.0 > -0.01:
                if i in self.right_indexes:
                    point = [point[0] + scale_factor * speed_vector[0], point[1], point[2]]
            elif speed_vector[0] < 0 and self.rotation[3] - 1.0 > -0.01:
                if i in self.left_indexes:
                    point = [point[0] + scale_factor * speed_vector[0], point[1], point[2]]
            elif speed_vector[1] > 0 and self.rotation[3] - 1.0 > -0.01:
                if i in self.front_indexes:
                    point = [point[0], point[1] + scale_factor * speed_vector[1], point[2]]
            elif speed_vector[1] < 0 and self.rotation[3] - 1.0 > -0.01:
                if i in self.back_indexes:
                    point = [point[0], point[1] + scale_factor * speed_vector[1], point[2]]
            elif speed_vector[0] > 0 and self.rotation[3] + 1.0 < 0.01:
                if i in self.left_indexes:
                    point = [point[0] + scale_factor * speed_vector[0], point[1], point[2]]
            elif speed_vector[0] < 0 and self.rotation[3] + 1.0 < 0.01:
                if i in self.right_indexes:
                    point = [point[0] + scale_factor * speed_vector[0], point[1], point[2]]
            elif speed_vector[1] > 0 and self.rotation[3] + 1.0 < 0.01:
                if i in self.back_indexes:
                    point = [point[0], point[1] + scale_factor * speed_vector[1], point[2]]
            elif speed_vector[1] < 0 and self.rotation[3] + 1.0 < 0.01:
                if i in self.front_indexes:
                    point = [point[0], point[1] + scale_factor * speed_vector[1], point[2]]
            
            scaled_points.append(point)        
                                        
            
        # Update the bounding box using the scaled points
        self.updateBoundingBox(scaled_points)
    
    def updateBoundingBox(self, points):
        # Update the bounding box using the new points
        for i in range(20):
            self.point_field.setMFVec3f(i, points[i])


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
