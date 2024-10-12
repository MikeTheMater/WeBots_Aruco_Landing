from controller import Supervisor, Emitter, Receiver, Node
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
        
        self.collision_Status= self.mavic.getField("customData").getSFString()
        
        self.nameDef = nameDef
    
    def change_bbox(self):
        speed_vector = self.mavic.getVelocity()[0:3]
        
        self.orientation = self.mavic.getOrientation()
        self.x_orientation = [self.orientation[0], self.orientation[3], self.orientation[6]]
        self.y_orientation = [self.orientation[1], self.orientation[4], self.orientation[7]]
        self.z_orientation = [self.orientation[2], self.orientation[5], self.orientation[8]]
        self.position = self.mavic.getPosition()
        # Normalize the speed vector
        speed_vector = speed_vector / np.linalg.norm(speed_vector) /2
        
        center=np.mean(self.points, axis=0)

        transformed_points = self.transform_points(self.points, self.orientation)
        
        # Calculate the normals of the points
        normals = Trying_the_normal.calculate_point_normals(transformed_points, self.triangles, center)
        threshold=0.2
        self.points_in_direction, self.points_opposite_direction = Trying_the_normal.classify_points_by_normal(normals, speed_vector, threshold)

        self.scaled_points = []
        for i in range(20):
            # Scale the points based on the speed in each direction
            point = self.points[i]

            speed_accuracy = 0.15 # speed accuracy to consider the drone is moving in a direction
            scale_factor = 1 # scale factor to move the points based on the speed vector
            #print(self.nameDef + " orientation ", self.orientation)   
            changed = False
            self.direction=0 #0 for x axis, 1 for -x axis, 2 for y axis, 3 for -y axis
            
            if self.position[2]> 0.1:
                if self.y_orientation[0] > - math.sqrt(2)/2 and  self.y_orientation[0] < math.sqrt(2)/2 and self.x_orientation[0] > math.sqrt(2)/2 :
                    if i in self.points_in_direction and not changed:
                        new_point = [point[0] + speed_vector[0] * scale_factor, point[1] + speed_vector[1] * scale_factor, point[2] + speed_vector[2] * scale_factor]
                        changed=True
                        self.direction = 0
                if self.y_orientation[0] > - math.sqrt(2)/2 and  self.y_orientation[0] < math.sqrt(2)/2 and self.x_orientation[0] < - math.sqrt(2)/2 :
                    if i in self.points_in_direction and not changed:
                        new_point = [point[0] - speed_vector[0] * scale_factor, point[1] - speed_vector[1] * scale_factor, point[2] + speed_vector[2] * scale_factor]
                        changed=True
                        self.direction = 1
                if self.x_orientation[0] > - math.sqrt(2)/2 and  self.x_orientation[0] < math.sqrt(2)/2 and self.y_orientation[0] < - math.sqrt(2)/2 :
                    if i in self.points_in_direction and not changed:
                        new_point = [point[0] + speed_vector[1] * scale_factor, point[1] - speed_vector[0] * scale_factor, point[2] + speed_vector[2] * scale_factor]
                        changed=True
                        self.direction = 2
                if self.x_orientation[0] > - math.sqrt(2)/2 and  self.x_orientation[0] < math.sqrt(2)/2 and self.y_orientation[0] > math.sqrt(2)/2 :
                    if i in self.points_in_direction and not changed:
                        new_point = [point[0] - speed_vector[1] * scale_factor, point[1] + speed_vector[0] * scale_factor, point[2] + speed_vector[2] * scale_factor]
                        changed=True
                        self.direction = 3
            
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
        triangles = self.get_triangles() 
        points_list = [vertex.tolist() for triangle in triangles for vertex in triangle]  # Convert to list of lists
        message = json.dumps({"name": self.nameDef, "position": position, "points": points_list})  # Serialize to JSON
        self.emitter.send(message.encode())  # Send as byte data

    def receive_data(self):
        self.other_drones_data.clear()
        while self.receiver.getQueueLength() > 0:
            received_message = self.receiver.getString()  # Get the message as a string
            data = json.loads(received_message)  # Deserialize JSON
            other_drone_name = data["name"]
            self.box2 = self.findPointsFromMessage(data["points"])  # Extract points
            self.other_drones_data.append((other_drone_name, self.box2))
            self.receiver.nextPacket()

            # Print points for both drones
            #print(f"Drone {self.nameDef} has these points: {self.points}")
            #print(f"Drone {other_drone_name} has these points: {self.box2}")
            
    def rotate_point(self, point, orientation_matrix):
        x_rot = np.dot(point, orientation_matrix[:3])  # Rotate the x component
        y_rot = np.dot(point, orientation_matrix[3:6])  # Rotate the y component
        z_rot = np.dot(point, orientation_matrix[6:9])  # Rotate the z component
        return [x_rot, y_rot, z_rot]

    def transform_points(self, points, orientation_matrix):
        return [self.rotate_point(point, orientation_matrix) for point in points]

    def check_collisions(self):
        self.box1 = self.get_triangles()
        for other_drone_name, other_triangles in self.other_drones_data:
            collision = self.findCollision(self.box1, other_triangles)
            if collision:
                print(f"Possible collision detected between {self.nameDef} and {other_drone_name}.")
                new_position = self.turn_right(1)
                self.mavic.getField("customData").setSFString(f"{new_position[0]} {new_position[1]} {new_position[2]}")
            else:
                #print(f"No collision detected between {self.nameDef} and {other_drone_name}.")
                #print(f"box of {self.nameDef}", self.box1)
                self.mavic.getField("customData").setSFString("0")

    def turn_right(self, distance):
        if self.direction == 0:
           # Move the drone 1 meter to the right
            new_position = [
                self.mavic.getPosition()[0],
                self.mavic.getPosition()[1] - distance,
                self.mavic.getPosition()[2]
            ]
        elif self.direction == 1:
            new_position = [
                self.mavic.getPosition()[0],
                self.mavic.getPosition()[1] + distance,
                self.mavic.getPosition()[2]
            ]
            
        elif self.direction == 2:
            # Move the drone 1 meter to the right
            new_position = [
                self.mavic.getPosition()[0] - distance,
                self.mavic.getPosition()[1],
                self.mavic.getPosition()[2]
            ]
        elif self.direction == 3:
            # Move the drone 1 meter to the right
            new_position = [
                self.mavic.getPosition()[0] + distance,
                self.mavic.getPosition()[1],
                self.mavic.getPosition()[2]
            ]
    
        # Set the new position (you may need to define a method to set position in your drone)
        #self.mavic.getField("translation").setSFVec3f(new_position)
        return new_position
    
    def findPointsFromMessage(self, points_list):
        points = [list(points_list[i:i+3]) for i in range(0, len(points_list), 3)]
        return points

    def get_triangles(self):
        # Get the current position of the drone
        position = self.mavic.getPosition()
        
        transformed_points = self.transform_points(self.scaled_points, self.orientation)
        # Convert self.triangles indices to actual coordinates from self.points
        triangles_coords = []
        for triangle in self.triangles:
            triangle_coords = np.array([transformed_points[idx] for idx in triangle])
            global_triangle_coords = triangle_coords + position  # Add position to each point
            triangles_coords.append(global_triangle_coords)
        return triangles_coords
    
    def findCollision(self, box1, box2, tolerance=1e-6):
        return box_intersection.boxes_intersect(box1, box2)
        
        
    def run(self):
        time_step=500
        while self.step(time_step) != -1:
            
            self.change_bbox()
            self.send_data()
            self.receive_data()
            self.check_collisions()
            self.simulationResetPhysics()
            
            if self.isNAN:
                break
            
            self.collision_Status= self.mavic.getField("customData").getSFString()
            if self.collision_Status=="landed":
                print("Landed")
                break
            
        print("Exiting...")
        sys.exit(0)