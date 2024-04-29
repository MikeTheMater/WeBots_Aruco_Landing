from mpl_toolkits.mplot3d import Axes3D

center=[-0.06, 0, 0.2]
dimensions=[0.47, 0.55, 0.1]

# Calculate half of the dimensions
half_dimensions = [dim / 2 for dim in dimensions]

# Calculate the 8 vertices of the box
vertices = [
    [round(center[dim] + half_dimensions[dim] * (1 if vertex & (1 << dim) else -1),2) for dim in range(3)]
    for vertex in range(8)
]
# Define the names for the vertices
names = ["back_bottom_left", "back_bottom_right", "front_bottom_left", "front_bottom_right",
         "back_top_left", "back_top_right", "front_top_left", "front_top_right"]

# Create a dictionary with the names and corresponding vertices
vertices_dict = {names[i]: vertices[i] for i in range(8)}

import matplotlib.pyplot as plt
# Define the connections between vertices to form the box
connections = [
    ("back_bottom_left", "back_bottom_right"),
    ("back_bottom_left", "front_bottom_left"),
    ("back_bottom_left", "back_top_left"),
    ("back_bottom_right", "front_bottom_right"),
    ("back_bottom_right", "back_top_right"),
    ("front_bottom_left", "front_bottom_right"),
    ("front_bottom_left", "front_top_left"),
    ("front_bottom_right", "front_top_right"),
    ("back_top_left", "back_top_right"),
    ("back_top_left", "front_top_left"),
    ("back_top_right", "front_top_right"),
    ("front_top_left", "front_top_right"),
]

# Calculate the center points of each line
center_points = []
center_point_names = []
new_connections = []
for connection in connections:
    point1 = vertices_dict[connection[0]]
    point2 = vertices_dict[connection[1]]
    center_point = [round((p1 + p2) / 2,2) for p1, p2 in zip(point1, point2)]
    center_points.append(center_point)
    center_point_names.append(connection[0] + "_" + connection[1] + "_center")
    new_connections.append((connection[0], connection[0] + "_" + connection[1] + "_center"))
    new_connections.append((connection[1], connection[0] + "_" + connection[1] + "_center"))


# Add the center points to the vertices and names
vertices.extend(center_points)
names.extend(center_point_names)

# Update the dictionary with the new vertices
vertices_dict = {names[i]: vertices[i] for i in range(len(names))}

# Print the connections
for connection in new_connections:
    index1 = names.index(connection[0])
    index2 = names.index(connection[1])
    print(f"Connect {connection[0]} (index {index1}) to {connection[1]} (index {index2})")

# Create a new figure
fig = plt.figure()

# Create a 3D plot
ax = fig.add_subplot(111, projection='3d')

# Unpack the vertices for plotting
x, y, z = zip(*vertices)



# Label the vertices
for i, name in enumerate(names):
    ax.text(x[i], y[i], z[i], str(name) + ' ' + str(i))
    # Plot the lines
    for connection in connections:
        x_values = [vertices_dict[connection[0]][0], vertices_dict[connection[1]][0]]
        y_values = [vertices_dict[connection[0]][1], vertices_dict[connection[1]][1]]
        z_values = [vertices_dict[connection[0]][2], vertices_dict[connection[1]][2]]
        ax.plot(x_values, y_values, z_values)

for vertex in vertices_dict:
    if vertices_dict[vertex][0] == -0.29:
        vertices_dict[vertex][0] = -0.3
    
    print(vertices_dict[vertex], vertex)
    
    # Define the directions
    directions = ["top", "bottom", "back", "front", "left", "right"]
    # Define the indexes for each direction
    indexes = {direction: [] for direction in directions}
    # Iterate over the vertices
    for name, vertex in vertices_dict.items():
        # Check if the vertex is on the top
        if vertex[2] == max(v[2] for v in vertices):
            indexes["top"].append(names.index(name))
        # Check if the vertex is on the bottom
        if vertex[2] == min(v[2] for v in vertices):
            indexes["bottom"].append(names.index(name))
        # Check if the vertex is on the back
        if vertex[1] == min(v[1] for v in vertices):
            indexes["back"].append(names.index(name))
        # Check if the vertex is on the front
        if vertex[1] == max(v[1] for v in vertices):
            indexes["front"].append(names.index(name))
        # Check if the vertex is on the left
        if vertex[0] == min(v[0] for v in vertices):
            indexes["left"].append(names.index(name))
        # Check if the vertex is on the right
        if vertex[0] == max(v[0] for v in vertices):
            indexes["right"].append(names.index(name))
# Print the indexes
for direction, index_list in indexes.items():
    print(f"The indexes of the points on the {direction} of the box are {index_list}")
    
# Plot the vertices
ax.scatter(x, y, z)
# Show the plot
plt.show()