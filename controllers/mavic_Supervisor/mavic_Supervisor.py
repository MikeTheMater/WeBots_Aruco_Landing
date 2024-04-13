from controller import Supervisor
import sys
import numpy as np
import math

class SuperMavic(Supervisor):
    def __init__(self):
        Supervisor.__init__(self)
        self.time_step = int(self.getBasicTimeStep())
        self.mavic = self.getFromDef("Mavic_2_PRO")
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
        self.initial_translation = self.pose_translation_field.getSFVec3f()
        print("Initial translation:", self.initial_translation)
        self.isNAN = False
        
    def calculateSpeed(self):
        time_step = int(self.getBasicTimeStep())
        while self.step(time_step) != -1:
            
            # Get the current position of the drone
            position1 = np.array([self.mavic.getPosition()[0], self.mavic.getPosition()[1], self.mavic.getPosition()[2]])

            # Wait for 1 second
            self.step(1000)

            # Get the updated position of the drone
            position2 = np.array([self.mavic.getPosition()[0], self.mavic.getPosition()[1], self.mavic.getPosition()[2]])

            # Calculate the difference in position
            position_difference = position2 - position1

            # Calculate the speed in each dimension
            speed = position_difference / 1.0  # Time difference is 1 second

            # Print the speed in each dimension
            print("Speed in x direction:", speed[0])
            print("Speed in y direction:", speed[1])
            print("Speed in z direction:", speed[2])
            return speed
    
    
    def change_bbox(self):
        speed_vector = self.calculateSpeed()
        center_of_mass = self.mavic.getCenterOfMass()
        print("Center of mass:", center_of_mass)
        
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
        
        # Update bounding box size based on normalized speed vector
        new_size = [self.initial_size[i] + abs(normalized_speed[i]) * scale_factor for i in range(3)]
        new_translation = [self.initial_translation[i] + (abs(normalized_speed[i]) if i < 2 else normalized_speed[i]) * scale_factor_position 
                            for i in range(3)]

        # Set the new size of the bounding box
        self.size_field.setSFVec3f(new_size)
        #Set the new translation of the bounding box
        self.pose_translation_field.setSFVec3f(new_translation)
        
        print("New size:", new_size)
        
    def run(self):
        while self.step(self.time_step) != -1:
            
            self.change_bbox()
            self.simulationResetPhysics()
            
            if self.isNAN:
               break
        print("Exiting...")
        sys.exit(0)

controller = SuperMavic()
controller.run()