import numpy as np

class Triangle:
    def __init__(self, a, b, c):
        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)

class Interval:
    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val

def cross(v1, v2):
    return np.cross(v1, v2)

def dot(v1, v2):
    return np.dot(v1, v2)

def get_interval(triangle, axis):
    """Get the projection of the triangle on the axis and return the interval."""
    proj1 = dot(triangle.a, axis)
    proj2 = dot(triangle.b, axis)
    proj3 = dot(triangle.c, axis)
    
    min_proj = min(proj1, proj2, proj3)
    max_proj = max(proj1, proj2, proj3)
    
    return Interval(min_proj, max_proj)

def overlap_on_axis(t1, t2, axis):
    """Check if the projections of the two triangles on the given axis overlap."""
    interval_a = get_interval(t1, axis)
    interval_b = get_interval(t2, axis)
    return (interval_b.min <= interval_a.max) and (interval_a.min <= interval_b.max)

def triangle_triangle(t1, t2):
    """Check if two triangles intersect using the separating axis theorem."""
    # Step 4: Compute the edges of triangle 1
    t1_f0 = t1.b - t1.a  # Edge 0 of triangle 1
    t1_f1 = t1.c - t1.b  # Edge 1 of triangle 1
    t1_f2 = t1.a - t1.c  # Edge 2 of triangle 1

    # Step 5: Compute the edges of triangle 2
    t2_f0 = t2.b - t2.a  # Edge 0 of triangle 2
    t2_f1 = t2.c - t2.b  # Edge 1 of triangle 2
    t2_f2 = t2.a - t2.c  # Edge 2 of triangle 2

    # Step 6: Build an array of potentially separating axes
    axis_to_test = [
        # Step 7: Normal of triangle 1
        cross(t1_f0, t1_f1),
        # Step 8: Normal of triangle 2
        cross(t2_f0, t2_f1),
        # Step 9: Cross products of edges of triangle 1 and triangle 2
        cross(t2_f0, t1_f0), cross(t2_f0, t1_f1), cross(t2_f0, t1_f2),
        cross(t2_f1, t1_f0), cross(t2_f1, t1_f1), cross(t2_f1, t1_f2),
        cross(t2_f2, t1_f0), cross(t2_f2, t1_f1), cross(t2_f2, t1_f2),
    ]

    # Step 10: Check for overlap on each axis
    for axis in axis_to_test:
        if np.linalg.norm(axis) < 1e-8:  # Skip degenerate axes
            continue
        if not overlap_on_axis(t1, t2, axis):
            return False  # Separating axis found, triangles do not intersect
    
    # Step 11: If no separating axis found, triangles intersect
    return True

# Function to check if two boxes intersect
def boxes_intersect(box1, box2):
    for triangle1_vertices in box1:
        triangle1 = Triangle(triangle1_vertices[0], triangle1_vertices[1], triangle1_vertices[2])
        for triangle2_vertices in box2:
            triangle2 = Triangle(triangle2_vertices[0], triangle2_vertices[1], triangle2_vertices[2])
            if triangle_triangle(triangle1, triangle2):
                return True
    return False