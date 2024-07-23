import numpy as np

def triangles_intersect(triangle1, triangle2, tolerance=1e-6):
    """Check if two triangles intersect with a given tolerance.

    Args:
        triangle1: np.array of shape (3, 3), representing the first triangle's vertices.
        triangle2: np.array of shape (3, 3), representing the second triangle's vertices.
        tolerance: float, the tolerance for floating-point comparisons.

    Returns:
        bool: True if the triangles intersect, False otherwise.
    """
    v1, v2, v3 = triangle1
    w1, w2, w3 = triangle2

    # Compute the normal vectors of the triangles
    n1 = np.cross(v2 - v1, v3 - v1)
    n2 = np.cross(w2 - w1, w3 - w1)

    # Check if the triangles are parallel
    if np.allclose(np.dot(n1, n2), 0, atol=tolerance):
        return False

    # Compute the distances from the origin to the triangles
    d1 = -np.dot(n1, v1)
    d2 = -np.dot(n2, w1)

    # Check if the triangles are on the same side of each other
    if (np.dot(n1, w1) + d1 < -tolerance and np.dot(n1, w2) + d1 < -tolerance and np.dot(n1, w3) + d1 < -tolerance) or \
       (np.dot(n2, v1) + d2 < -tolerance and np.dot(n2, v2) + d2 < -tolerance and np.dot(n2, v3) + d2 < -tolerance):
        return False

    # Check for intersection along the edges of the triangles
    for (a, b) in [(v1, v2), (v2, v3), (v3, v1)]:
        for (c, d) in [(w1, w2), (w2, w3), (w3, w1)]:
            if edges_intersect(a, b, c, d, tolerance):
                return True

    return False

def edges_intersect(a, b, c, d, tolerance=1e-6):
    """Check if two edges (a, b) and (c, d) intersect with a given tolerance.

    Args:
        a, b: np.array of shape (3,), representing the first edge's endpoints.
        c, d: np.array of shape (3,), representing the second edge's endpoints.
        tolerance: float, the tolerance for floating-point comparisons.

    Returns:
        bool: True if the edges intersect, False otherwise.
    """
    def ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) > (q[1] - p[1]) * (r[0] - p[0])

    if ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d):
        return True

    return False


def is_point_in_triangle(pt, tri):
    """Check if a point is inside a triangle.

    Args:
        pt: np.array of shape (3,), the point to check.
        tri: np.array of shape (3, 3), the triangle vertices.

    Returns:
        bool: True if the point is inside the triangle, False otherwise.
    """
    v2 = pt - tri[0]
    v0 = tri[1] - tri[0]
    v1 = tri[2] - tri[0]
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom
    return (u >= 0) and (v >= 0) and (u + v < 1)
