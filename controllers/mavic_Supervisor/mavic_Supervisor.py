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
        self.bounding_object = self.mavic.getField("boundingObject").getSFNode()
        self.children_field = self.bounding_object.getField("children")
        self.first_pose_node = self.children_field.getMFNode(0)
        self.pose_children_field = self.first_pose_node.getField("children")
        self.bounding_box = self.pose_children_field.getMFNode(0)
        self.size_field = self.bounding_box.getField("size")
        self.initial_size = [0.47, 0.55, 0.1]
        self.size_field.setSFVec3f(self.initial_size)
        self.pose_translation_field = self.first_pose_node.getField("translation")
        self.initial_translation = self.pose_translation_field.getSFVec3f()#the initial translation of the bounding box in relation to the drone
        #self.initial_translation = self.calculate_translation_rel_to_world(self.initial_translation)
        print("Initial translation:", self.initial_translation)
        self.isNAN = False
        
        self.mavic.getEmitter = Emitter("emitter")
        self.mavic.getReceiver = Receiver("receiver")
        self.emitter = self.mavic.getEmitter
        self.receiver = self.mavic.getReceiver
        
        self.nameDef = nameDef
    
    def calculate_translation_rel_to_world(self, translation):
        return [translation[0] + self.mavic.getPosition()[0], translation[1] + self.mavic.getPosition()[1], translation[2] + self.mavic.getPosition()[2]]
    
    def calculateSpeed(self):
        time_step = int(self.getBasicTimeStep())
        while self.step(time_step) != -1:
            
            # Get the current position of the drone
            position1 = np.array([self.mavic.getPosition()[0], self.mavic.getPosition()[1], self.mavic.getPosition()[2]])

            # Wait for 1 second
            self.step(500)

            # Get the new position of the drone
            position2 = np.array([self.mavic.getPosition()[0], self.mavic.getPosition()[1], self.mavic.getPosition()[2]])

            # Calculate the difference in position
            position_difference = position2 - position1

            # Calculate the speed in each dimension
            speed = position_difference / 0.5  # Time difference is 1 second

            # Print the speed in each dimension (in meters per second)
            #print("Speed in x direction:", speed[0])
            #print("Speed in y direction:", speed[1])
            #print("Speed in z direction:", speed[2])
            return speed
    
    
    def change_bbox(self):
        speed_vector = self.calculateSpeed()
        
        # Normalize the speed vector
        speed_magnitude = np.linalg.norm(speed_vector)
        if speed_magnitude == 0:
            return  # Skip updating bounding box if speed is zero
        normalized_speed = speed_vector / speed_magnitude
        #print("Normalized speed vector:", normalized_speed)
        
        
        #speed scale factor
        scale_factor = 1
        #bounding box position scale factor
        scale_factor_position = 0.5
        
        # Update bounding box size and center based on normalized speed vector
        self.new_size = [self.initial_size[i] + abs(normalized_speed[i]) * scale_factor for i in range(3)]
        self.new_translation = [self.initial_translation[i] + (abs(normalized_speed[i]) if i < 2 else normalized_speed[i]) * scale_factor_position 
                            for i in range(3)]

        position= self.mavic.getPosition()
        if position[2]<self.new_size[2] and normalized_speed[2]<0:
            self.new_size[2]=self.initial_size[2]
            self.new_translation[2]=self.initial_translation[2]    
        
        # Set the new size of the bounding box
        self.size_field.setSFVec3f(self.new_size)
        #Set the new translation of the bounding box
        self.pose_translation_field.setSFVec3f(self.new_translation)
    
    def calculateSpaceOfBox(self):
        position = self.mavic.getPosition()
        # Calculate the 8 different points of the bounding box
        points = []
        rounding_factor = 3
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    x = position[0] + (-1) ** i * self.new_size[0] / 2
                    y = position[1] + (-1) ** j * self.new_size[1] / 2
                    z = position[2] + (-1) ** k * self.new_size[2] / 2
                    points.append((round(x, rounding_factor), round(y, rounding_factor), round(z, rounding_factor)))

        # Rearrange the points in the required order for the Box class
        box_vertices = [
            points[0],
            points[1],
            points[3],
            points[2],
            points[4],
            points[5],
            points[7],
            points[6]
        ]
        return box_vertices

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
            message="bbox of "+self.nameDef+" "+str(self.calculateSpaceOfBox())
            self.emitter.send(message)
            # Example: Receive a message on the receiver
            if self.receiver.getQueueLength() > 0:
                received_message = self.receiver.getString()
                #print("Received message to "+self.nameDef+":" , received_message)
                self.receiver.nextPacket()  # Move to the next received packet

                box1=self.calculateSpaceOfBox()                

                
                box2 = self.findPointsFromMessage(received_message)
                
                collision = self.findCollision(box1, box2)
                if collision:
                    print("Collision detected from " + self.nameDef + " with the other drone.")
                    # Handle collision logic here
                else:
                    print("No collision detected from " + self.nameDef + " with the other drone.")

            
            self.simulationResetPhysics()
            
            if self.isNAN:
               break
        #     if self.mavic.getPosition()[2] < 0.1 and self.calculateSpeed()[2] < 0:
        #         print("Landed")
        #         break
        # print("Exiting...")
        sys.exit(0)

# controller1 = SuperMavic("Mavic_2_PRO")
# #controller2 = SuperMavic("Mavic_2_PRO_2")
# controller1.run()