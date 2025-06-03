import numpy as np
from itertools import product

# -----------------------------
# Geometry Classes
# -----------------------------

class Cylinder:
    def __init__(self, p1, p2, r):
        """
        p1: np.array([x, y, z]) — bottom center of the cylinder
        p2: np.array([x, y, z]) — top center of the cylinder
        r: float — radius
        """
        self.p1 = np.array(p1, dtype=float)
        self.p2 = np.array(p2, dtype=float)
        self.r = float(r)


class Box:
    def __init__(self, center, size):
        """
        center: np.array([x, y, z]) — center of the box
        size: np.array([sx, sy, sz]) — dimensions of the box
        """
        self.center = np.array(center, dtype=float)
        self.size = np.array(size, dtype=float)

    def bounds(self):
        half = self.size / 2
        return self.center - half, self.center + half


# -----------------------------
# Collision Detection Function
# -----------------------------

def check_collision(cylinder: Cylinder, box: Box) -> bool:
    P1, P2, r = cylinder.p1, cylinder.p2, cylinder.r
    box_min, box_max = box.bounds()

    # Step 1: AABB reject
    cyl_min = np.minimum(P1, P2) - r
    cyl_max = np.maximum(P1, P2) + r
    if np.any(cyl_max < box_min) or np.any(cyl_min > box_max):
        return False

    # Step 2: Check if cylinder axis intersects box
    if segment_intersects_aabb(P1, P2, box_min, box_max):
        return True

    # Step 3: Check edges
    edges, vertices = get_box_edges_and_vertices(box.center, box.size)
    for p, q in edges:
        if segment_segment_distance(P1, P2, p, q) <= r:
            return True

    # Step 4: Check vertices
    for v in vertices:
        if point_to_segment_distance(v, P1, P2) <= r:
            return True

    return False


# -----------------------------
# Utility Functions
# -----------------------------

def segment_intersects_aabb(P1, P2, box_min, box_max):
    tmin, tmax = 0.0, 1.0
    d = P2 - P1
    for i in range(3):
        if abs(d[i]) < 1e-8:
            if P1[i] < box_min[i] or P1[i] > box_max[i]:
                return False
        else:
            ood = 1.0 / d[i]
            t1 = (box_min[i] - P1[i]) * ood
            t2 = (box_max[i] - P1[i]) * ood
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return False
    return True

def get_box_edges_and_vertices(center, size):
    half = size / 2
    box_min = center - half
    box_max = center + half
    corners = [np.array([x, y, z]) for x, y, z in product(
        [box_min[0], box_max[0]],
        [box_min[1], box_max[1]],
        [box_min[2], box_max[2]]
    )]
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.count_nonzero(np.abs(corners[i] - corners[j])) == 1:
                edges.append((corners[i], corners[j]))
    return edges, corners

def segment_segment_distance(p1, q1, p2, q2):
    u = q1 - p1
    v = q2 - p2
    w = p1 - p2
    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w)
    e = np.dot(v, w)
    D = a * c - b * b
    SMALL_NUM = 1e-8

    if D < SMALL_NUM:
        sc = 0.0
        tc = e / c if c > SMALL_NUM else 0.0
    else:
        sc = (b * e - c * d) / D
        tc = (a * e - b * d) / D

    sc = np.clip(sc, 0.0, 1.0)
    tc = np.clip(tc, 0.0, 1.0)

    dP = w + sc * u - tc * v
    return np.linalg.norm(dP)

def point_to_segment_distance(p, a, b):
    ab = b - a
    t = np.dot(p - a, ab) / np.dot(ab, ab)
    t = np.clip(t, 0, 1)
    projection = a + t * ab
    return np.linalg.norm(p - projection)
