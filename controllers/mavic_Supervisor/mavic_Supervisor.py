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
        self.initial_size = self.size_field.getSFVec3f()
        self.isNAN = False
        
    def change_bbox(self):
        speed_vector=self.mavic.getVelocity()
        print("Speed vector:" , speed_vector)
        add = 0
        if math.isnan(speed_vector[0 + add ]) or math.isnan(speed_vector[1 + add ]) or math.isnan(speed_vector[2 + add]):
            print("Speed vector is NaN, changing it to 0.")
            speed_vector = [0,0,0]
            self.isNAN = True
            return
        self.size_field.setSFVec3f([self.initial_size[0] + abs(speed_vector[0 + add]), self.initial_size[1] + abs(speed_vector[1 + add ]), self.initial_size[2] + abs(speed_vector[2 + add])])
    
        
    def run(self):
        while self.step(self.time_step) != -1:
            self.change_bbox()
            if self.isNAN:
               break

controller = SuperMavic()
controller.run()