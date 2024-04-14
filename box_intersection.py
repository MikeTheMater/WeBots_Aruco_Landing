import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Box:
    def __init__(self, vertices):
        self.vertices = np.array(vertices)

    def is_point_inside(self, point):
        x, y, z = point
        xmin, ymin, zmin = self.vertices[0]
        xmax, ymax, zmax = self.vertices[7]
        return xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax

def boxes_intersect(box1, box2):
    for vertex in box1.vertices:
        if box2.is_point_inside(vertex):
            return True
    for vertex in box2.vertices:
        if box1.is_point_inside(vertex):
            return True
    return False

def plot_box(ax, vertices, color='b'):
    edges = [
        [vertices[0], vertices[1], vertices[2], vertices[3], vertices[0]],
        [vertices[4], vertices[5], vertices[6], vertices[7], vertices[4]],
        [vertices[0], vertices[4]],
        [vertices[1], vertices[5]],
        [vertices[2], vertices[6]],
        [vertices[3], vertices[7]]
    ]
    for edge in edges:
        ax.plot3D(*zip(*edge), color=color)

def example_run():
    # Example usage:
    box1_vertices = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    box2_vertices = [(0, 0, 0), (1.5, 0.5, 0.5), (1.5, 1.5, 0.5), (0.5, 1.5, 0.5),
                    (0.5, 0.5, 1.5), (1.5, 0.5, 1.5), (1.5, 1.5, 1.5), (0.5, 1.5, 1.5)]

    box1 = Box(box1_vertices)
    box2 = Box(box2_vertices)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot_box(ax, box1.vertices, color='b')
    plot_box(ax, box2.vertices, color='r')

    if boxes_intersect(box1, box2):
        print("The boxes intersect.")
    else:
        print("The boxes do not intersect.")

    plt.show()

def get_points_from_message():
    # Assuming message is constructed as described
    message = "bbox of Mavic_2_PRO_2 [(0.18, 0.28, 1.03), (0.18, 0.28, -0.07), (0.18, -0.28, -0.07), (0.18, -0.28, 1.03), (-0.3, 0.28, 1.03), (-0.3, 0.28, -0.07), (-0.3, -0.28, -0.07), (-0.3, -0.28, 1.03)]"

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

    print(vertices_list)
    # Combine the vertices into triples
    points = [tuple(vertices_list[i:i+3]) for i in range(0, len(vertices_list), 3)]

    print(points)
    # Convert each string element to a float and keep it in a tuple
    box_vertices = tuple(float(coord) for coord in vertices_list)

    print(box_vertices)
