from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

# Function to compute the normal of a triangle given its 3 vertices
def compute_triangle_normal(v1, v2, v3, reference_point):
    # Compute the vectors of two edges of the triangle
    edge1 = np.array(v2) - np.array(v1)
    edge2 = np.array(v3) - np.array(v1)
    
    # Compute the normal as the cross product of the two edge vectors
    normal = np.cross(edge1, edge2)
    
    # Normalize the normal vector
    norm_length = np.linalg.norm(normal)
    if norm_length != 0:
        normal = normal / norm_length

    # Ensure the normal is pointing outward
    centroid = np.mean([v1, v2, v3], axis=0)
    if np.dot(centroid - reference_point, normal) < 0:
        normal = -normal
    
    return normal

# Function to calculate normals for each point by averaging normals of triangles it is part of
def calculate_point_normals(points, triangles, reference_point):
    # Initialize zero vectors for each point's normal
    normals = [np.zeros(3) for _ in range(len(points))]
    
    for triangle in triangles:
        # Get the vertices of the triangle
        v1, v2, v3 = points[triangle[0]], points[triangle[1]], points[triangle[2]]
        
        # Calculate the normal of the triangle
        normal = compute_triangle_normal(v1, v2, v3, reference_point)
        
        # Add this normal to each of the triangle's vertices
        normals[triangle[0]] += normal
        normals[triangle[1]] += normal
        normals[triangle[2]] += normal
    
    # Normalize each point's normal vector
    for i, n in enumerate(normals):
        norm_length = np.linalg.norm(n)
        if norm_length != 0:
            normals[i] = n / norm_length
    
    return normals

# Function to classify points based on their normals and the speed vector
def classify_points_by_normal(normals, speed_vector, dot_product_threshold):
    # Normalize the speed vector
    speed_norm = speed_vector / np.linalg.norm(speed_vector)
    
    points_in_direction = []
    points_opposite_direction = []
    
    # Classify each point based on the dot product of its normal and the speed vector
    for i, normal in enumerate(normals):
        dot_product = np.dot(normal, speed_norm)
        #print(f"Point {i} Normal: {normal}, Dot Product: {dot_product}")  # Print the normal and dot product
        
        if dot_product >= dot_product_threshold:
            points_in_direction.append(i)
        else:
            points_opposite_direction.append(i)
    
    return points_in_direction, points_opposite_direction

def find_triangles(points, coordIndex):
    # Filter out the -1 values and create pairs of connections
    connections = [(coordIndex[i], coordIndex[i + 1]) for i in range(0, len(coordIndex), 3) if coordIndex[i + 2] == -1]
    # Find triangles
    triangles = []

    triangles_to_ignore = [[10,16,17], [14,17,19], [15,18,19], [12,16,18], [8,9,10], [8,11,12], [11,13,15], [9,13,14]]

    for i, (start1, end1) in enumerate(connections):
        for j, (start2, end2) in enumerate(connections):
            if i != j:
                if end1 == start2:
                    for k, (start3, end3) in enumerate(connections):
                        if k != i and k != j:
                            if end2 == start3 and end3 == start1:
                                triangle = sorted([start1, end1, end2])
                                if len(set(triangle)) == 3 and triangle not in triangles and triangle not in triangles_to_ignore:
                                    triangles.append(triangle)
                elif end1 == end2:
                    for k, (start3, end3) in enumerate(connections):
                        if k != i and k != j:
                            if start2 == start3 and start1 == end3:
                                triangle = sorted([start1, end1, end2])
                                if len(set(triangle)) == 3 and triangle not in triangles and triangle not in triangles_to_ignore:
                                    triangles.append(triangle)
                elif start1 == start2:
                    for k, (start3, end3) in enumerate(connections):
                        if k != i and k != j:
                            if end2 == end3 and end1 == start3:
                                triangle = sorted([start1, end1, end2])
                                if len(set(triangle)) == 3 and triangle not in triangles and triangle not in triangles_to_ignore:
                                    triangles.append(triangle)
                elif start1 == end2:
                    for k, (start3, end3) in enumerate(connections):
                        if k != i and k != j:
                            if start2 == end3 and start3 == end1:
                                triangle = sorted([start1, end1, end2])
                                if len(set(triangle)) == 3 and triangle not in triangles and triangle not in triangles_to_ignore:
                                    triangles.append(triangle)
    return triangles


# Convert the points list into a list of lists
# points = [
#     [-0.3, -0.28, 0.15], [0.17, -0.28, 0.15], [-0.3, 0.28, 0.15], [0.17, 0.28, 0.15],
#     [-0.3, -0.28, 0.25], [0.17, -0.28, 0.25], [-0.3, 0.28, 0.25], [0.17, 0.28, 0.25],
#     [-0.06, -0.28, 0.15], [-0.3, 0, 0.15], [-0.3, -0.28, 0.2], [0.17, 0, 0.15],
#     [0.17, -0.28, 0.2], [-0.06, 0.28, 0.15], [-0.3, 0.28, 0.2], [0.17, 0.28, 0.2],
#     [-0.06, -0.28, 0.25], [-0.3, 0, 0.25], [0.17, 0, 0.25], [-0.06, 0.28, 0.25]
# ]
# # List of indexes indicating connections between points
# coordIndex = [
#     0, 8, -1, 1, 8, -1, 0, 9, -1, 2, 9, -1, 0, 10, -1, 4, 10, -1, 1, 11, -1, 
#     3, 11, -1, 1, 12, -1, 5, 12, -1, 2, 13, -1, 3, 13, -1, 2, 14, -1, 6, 14, -1, 
#     3, 15, -1, 7, 15, -1, 4, 16, -1, 5, 16, -1, 4, 17, -1, 6, 17, -1, 5, 18, -1, 
#     7, 18, -1, 6, 19, -1, 7, 19, -1, 8, 10, -1, 8, 12, -1, 16, 10, -1, 16, 12, -1, 
#     8, 16, -1, 9, 14, -1, 9, 10, -1, 17, 14, -1, 17, 10, -1, 16, 17, -1, 16, 18, -1, 
#     19, 17, -1, 19, 18, -1, 17, 18, -1, 8, 9, -1, 8, 11, -1, 13, 9, -1, 13, 11, -1, 
#     9, 11, -1, 13, 15, -1, 13, 14, -1, 19, 15, -1, 19, 14, -1, 13, 19, -1, 10, 17, -1, 
#     10, 9, -1, 14, 17, -1, 14, 9, -1, 9, 17, -1, 12, 18, -1, 12, 11, -1, 15, 18, -1, 
#     15, 11, -1, 18, 11, -1
# ]