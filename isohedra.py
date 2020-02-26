import numpy as np
from scipy.spatial.distance import cdist

def load_object(filename):
    """Load polyhedron model from minimal Wavefront OBJ file."""
    vertices = []
    faces = []
    with open(filename) as file:
        for line in file:
            words = line.split()
            if len(words) > 0 and words[0] in ['v', 'f']:
                if words[0] == 'v':
                    vertices.append([float(w) for w in words[1:]])
                else:
                    faces.append([int(w) - 1 for w in words[1:]])
    return vertices, faces

def symmetries(vertices, face1, face2, full=True, v_tol=1e-8):
    """Generate convex polyhedron symmetries mapping face1 to face2."""

    # Columns of A are consecutive vertices of face1
    # (convexity and non-degenerate edges guarantee these are not coplanar).
    A = np.transpose([vertices[face1[i]] for i in range(3)])
    m = len(face2)

    # Walk forward and backward around face2.
    for det in [1, -1]:
        for i0 in range(m):

            # Columns of B are consecutive vertices of face2.
            B = np.transpose([vertices[face2[(i0 + det * i) % m]]
                              for i in range(3)])

            # Matrix R maps (3 vertices of) face1 to face2; check if all other
            # vertices match up as well.
            R = B @ np.linalg.inv(A)
            if full or np.linalg.det(R) > 0.5:
                distances = cdist(vertices, vertices @ np.transpose(R))
                if max([min(d) for d in distances]) <= v_tol:
                    yield R

def symmetry_group(vertices, faces, full=True, v_tol=1e-8, r_tol=1e-3):
    """Return full/proper symmetry group of given convex polyhedron."""
    group = set()
    for face1 in faces:
        for face2 in faces:
            for R in symmetries(vertices, face1, face2, full, v_tol):
                if min((abs(R - g).max() for g in group),
                       default=float('Inf')) > r_tol:
                    group.add(tuple(tuple(row) for row in R))
    return group

def face_orbits(vertices, faces, v_tol=1e-8):
    """Return orbits of proper symmetry group action on polyhedron faces."""
    classes = []
    for face in faces:
        found = False
        for c in classes:
            if len(list(symmetries(vertices, face, c[0], False, v_tol))) > 0:
                c.append(face)
                found = True
                break
        if not found:
            classes.append([face])
    return classes

if __name__ == '__main__':
    import os
    for filename in os.listdir('models'):
        vertices, faces = load_object('models/' + filename)
        print(filename,
              len(symmetry_group(vertices, faces)),
              len(symmetry_group(vertices, faces, False)),
              len(face_orbits(vertices, faces)))
