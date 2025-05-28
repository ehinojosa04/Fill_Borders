# Computational Geometry final project 2025
import numpy as np
import matplotlib.pyplot as plt


# read obj file
def readObjFile(file_path):
    vertices = []
    faces = []
    with open(file_path, 'r') as obj_file:
        for line in obj_file:
            if line.startswith('v '):
                vertex = list(map(float, line.strip().split()[1:]))
                vertices.append(vertex)
            elif line.startswith('f '):
                face = list(map(int, line.strip().split()[1:]))
                face = [index - 1 for index in face]
                faces.append(face)
    return vertices, faces

# Half-edge data structure
class Vertex:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.halfEdge = None

class Face:
    def __init__(self):
        self.halfEdge = None

class HalfEdge:
    def __init__(self):
        self.origin = None
        self.twin = None
        self.face = None
        self.next = None
        self.prev = None

def VFtoHEDS(vertices, faces):
    verticesArray = {i: Vertex(*v) for i, v in enumerate(vertices)}
    halfEdgesArray = []
    facesArray = {}
    edge_dict = {}

    for face_index, face in enumerate(faces):
        f = Face()
        facesArray[face_index] = f

        face_halfedges = []
        for i in range(3):
            he = HalfEdge()
            he.origin = verticesArray[face[i]]
            verticesArray[face[i]].halfEdge = he
            he.face = f
            face_halfedges.append(he)
            halfEdgesArray.append(he)

        for i in range(3):
            face_halfedges[i].next = face_halfedges[(i + 1) % 3]
            face_halfedges[i].prev = face_halfedges[(i + 2) % 3]

            origin = face[i]
            dest = face[(i + 1) % 3]
            edge_dict[(origin, dest)] = face_halfedges[i]

        f.halfEdge = face_halfedges[0]

    for (origin, dest), he in edge_dict.items():
        twin = edge_dict.get((dest, origin))
        if twin:
            he.twin = twin

    return verticesArray, halfEdgesArray, facesArray

def getFaceVertices(face):
    vertices = []
    start_halfEdge = face.halfEdge
    current_halfEdge = start_halfEdge
    while True:
        vertices.append(current_halfEdge.origin)
        current_halfEdge = current_halfEdge.next
        if current_halfEdge == start_halfEdge:
            break
    return vertices

def HEDStoVF(verticesArray, halfEdgesArray, facesArray):
    vertices = [(v.x, v.y, v.z) for v in verticesArray.values()]
    faces = []
    for face in facesArray.values():
        face_vertices = getFaceVertices(face)
        face_indices = [list(verticesArray.keys())[list(verticesArray.values()).index(v)] for v in face_vertices]
        faces.append(face_indices)
    return vertices, faces

def writeObjFile(vertices, faces, output_file):
    with open(output_file, 'w') as obj_file:
        for vertex in vertices:
            obj_file.write('v ' + ' '.join(map(str, vertex)) + '\n')
        for face in faces:
            obj_file.write('f ' + ' '.join(map(lambda x: str(x + 1), face)) + '\n')

def visualizeMesh(vertices, faces, holes=None, filled_faces=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    vertices_array = np.array(vertices)
    ax.scatter(vertices_array[:, 0], vertices_array[:, 2], vertices_array[:, 1], c='k', depthshade=False)
    
    for face in faces:
        face_vertices = [vertices[i] for i in face]
        face_vertices.append(vertices[face[0]])
        face_vertices = np.array(face_vertices)
        ax.plot(face_vertices[:, 0], face_vertices[:, 2], face_vertices[:, 1], c='b')

    if holes:
        for hole in holes:
            x = [v.x for v in hole] + [hole[0].x]
            y = [v.y for v in hole] + [hole[0].y]
            z = [v.z for v in hole] + [hole[0].z]
            ax.plot(x, z, y, color='r', linewidth=2)

    if filled_faces:
        for face in filled_faces:
            verts = getFaceVertices(face)
            x = [v.x for v in verts] + [verts[0].x]
            y = [v.y for v in verts] + [verts[0].y]
            z = [v.z for v in verts] + [verts[0].z]
            ax.plot(x, z, y, color='g', linewidth=2)

    ax.set_box_aspect([np.ptp(vertices_array[:, 0]), np.ptp(vertices_array[:, 0]), np.ptp(vertices_array[:, 1])])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Mesh with Triangular Holes Filled (Green)")
    plt.show()


def detect_holes(halfEdgesArray, ignore_largest=False):
    visited = set()
    all_loops = []

    for he in halfEdgesArray:
        if he.twin is None and he not in visited:
            loop = []
            current = he
            while current not in visited:
                visited.add(current)
                loop.append(current.origin)
                current = current.next
                while current.twin:
                    current = current.twin.next
            all_loops.append(loop)
            
    print(len(all_loops), "holes detected")

    if ignore_largest and len(all_loops) > 1:
        all_loops.sort(key=lambda loop: -len(loop))
        return all_loops[1:]
    return all_loops

def draw_pca_plane(hole):
    """
    Given a list of Vertex objects forming a hole boundary,
    draw the best-fit PCA plane and the projected points.
    """
    # Convert to numpy array
    hole_points = np.array([[v.x, v.y, v.z] for v in hole])

    # PCA
    centroid = np.mean(hole_points, axis=0)
    centered = hole_points - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    normal = eigvecs[:, 0]
    u = eigvecs[:, 1]
    v = eigvecs[:, 2]

    # Project to 2D and back to 3D
    proj_2d = np.array([[np.dot(p, u), np.dot(p, v)] for p in centered])
    projected_back = np.array([centroid + x*u + y*v for x, y in proj_2d])

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Original points
    ax.scatter(hole_points[:,0], hole_points[:,1], hole_points[:,2], color='blue', label='Original 3D points')

    # Projected points on plane
    ax.scatter(projected_back[:,0], projected_back[:,1], projected_back[:,2], color='green', label='Projected points (on plane)')

    # Plane mesh
    plane_size = 2
    plane_grid_u, plane_grid_v = np.meshgrid(np.linspace(-plane_size, plane_size, 10), np.linspace(-plane_size, plane_size, 10))
    plane_points = centroid[:, np.newaxis, np.newaxis] + u[:, np.newaxis, np.newaxis]*plane_grid_u + v[:, np.newaxis, np.newaxis]*plane_grid_v
    ax.plot_surface(plane_points[0], plane_points[1], plane_points[2], alpha=0.3, color='gray')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('PCA Plane and Projected Points')
    plt.tight_layout()
    plt.show()
    return True


def process_mesh(input_name: str, output_name: str, ignore_outer_boundary=True):
    vertices, faces = readObjFile(input_name)
    verticesArray, halfEdgesArray, facesArray = VFtoHEDS(vertices, faces)
    
    holes = detect_holes(halfEdgesArray, ignore_largest= not ignore_outer_boundary)

    filled_faces = []

    #Fill triangular holes
    for hole in holes:
        if len(hole) == 3:
            # Create new face
            f = Face()
            facesArray[len(facesArray)] = f
            filled_faces.append(f)

            new_halfedges = []
            for i in range(3):
                he = HalfEdge()
                he.origin = hole[i]
                he.face = f
                new_halfedges.append(he)
                halfEdgesArray.append(he)

            # Link next/prev
            for i in range(3):
                new_halfedges[i].next = new_halfedges[(i + 1) % 3]
                new_halfedges[i].prev = new_halfedges[(i - 1) % 3]

            f.halfEdge = new_halfedges[0]

            # Assign twins
            for i in range(3):
                origin = hole[i]
                dest = hole[(i + 1) % 3]
                for he in halfEdgesArray:
                    if he.origin == dest and he.next.origin == origin:
                        he.twin = new_halfedges[i]
                        new_halfedges[i].twin = he
                        break
        else:
            #Fill non-triangular meshes
            hole_points = np.array([(v.x, v.y, v.z) for v in hole])
            
            centroid = np.mean(hole_points, axis=0)
            centered = hole_points - centroid
            
            cov = np.cov(centered.T)

            eigvals, eigvecs = np.linalg.eigh(cov)
            
            u = eigvecs[:, 1]
            v = eigvecs[:, 2]
            
            # Project centered points to the plane
            proj_2d = np.array([
                [np.dot(p, u), np.dot(p, v)]
                for p in centered
            ])
                        
        
    newVertices, newFaces = HEDStoVF(verticesArray, halfEdgesArray, facesArray)
    writeObjFile(newVertices, newFaces, output_name)
    visualizeMesh(newVertices, newFaces, holes, filled_faces)

