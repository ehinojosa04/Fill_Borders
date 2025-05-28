
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

# plot the mesh (optional, we will use meshlab) 

# convert to HEDS (half-edges data structure)
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
    # create vertices
    verticesArray = {}
    for i, (x, y, z) in enumerate(vertices):
        vertex = Vertex(x, y, z)
        verticesArray[i] = vertex

    # Create half edges
    halfEdgesArray = [HalfEdge() for _ in range(len(faces)*3)]

    # Create faces
    facesArray = {}
    for i, _ in enumerate(faces):
        face = Face()
        facesArray[i] = face

    # Connect halfEdges to vertices
    for face_index, face_vertices in enumerate(faces): 
        for i, vertex_index in enumerate(face_vertices):
            halfEdge = halfEdgesArray[face_index *3 +i]
            halfEdge.vertex = verticesArray[vertex_index]
            verticesArray[vertex_index].halfEdge = halfEdge
    
    # Connect half edges to faces
    for face_index, face_vertices in enumerate(faces):
        face = facesArray[face_index]
        face_halfEdgesArray = [halfEdgesArray[face_index * 3 + i] for i in range(3)]
        face.halfEdge = face_halfEdgesArray[0]
        for i in range(3):
            face_halfEdgesArray[i].face = face
            face_halfEdgesArray[i].next = face_halfEdgesArray[(i + 1) % 3]
            face_halfEdgesArray[i].prev = face_halfEdgesArray[(i + 2) % 3]

    # Connect twins

    return verticesArray, halfEdgesArray, facesArray






# convert back to VF (Vertices Faces data structure)
def getFaceVertices(face):
    vertices = []
    start_halfEdge = face.halfEdge
    current_halfEdge = start_halfEdge
    while True:
        vertices.append(current_halfEdge.vertex)
        current_halfEdge = current_halfEdge.next
        if current_halfEdge == start_halfEdge:
            break
    return vertices

def HEDStoVF(verticesArray, halfEdgesArray, facesArray):
    vertices = [(vertex.x, vertex.y, vertex.z) for vertex in verticesArray.values()]
    faces = []
    for face in facesArray.values():
        face_vertices = getFaceVertices(face)
        face_indices = [list(verticesArray.keys())[list(verticesArray.values()).index(vertex)] for vertex in face_vertices]
        faces.append(face_indices)
    return vertices, faces

# write the final .obj or .ply file (with color coding)
def writeObjFile(vertices, faces, output_file):
    with open(output_file, 'w') as obj_file:
        for vertex in vertices:
            obj_file.write('v ' + ' '.join(map(str, vertex)) + '\n')
        for face in faces:
            obj_file.write('f ' + ' '.join(map(lambda x: str(x + 1), face)) + '\n')

def visualizeMesh(vertices, faces, vertex_color='k', edge_color='b'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot vertices
    vertices_array = np.array(vertices)
    ax.scatter(vertices_array[:,0], vertices_array[:,2], vertices_array[:,1], c=vertex_color, depthshade=False)
    
    # Plot edges
    for face in faces:
        face_vertices = [vertices[i] for i in face]
        face_vertices.append(vertices[face[0]])  # Close the loop
        face_vertices = np.array(face_vertices)
        ax.plot(face_vertices[:,0], face_vertices[:,2], face_vertices[:,1], c=edge_color)  
        
    # Set equal aspect ratio
    ax.set_box_aspect([np.ptp(vertices_array[:,0]), np.ptp(vertices_array[:,0]), np.ptp(vertices_array[:,1])])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()
    
def process_mesh(input_name: str, output_name: str):
    # read obj
    vertices, faces = readObjFile(input_name)

    # visualize the mesh
    visualizeMesh(vertices, faces)   

    # convert to HEDS
    verticesArray, halfEdgesArray, facesArray = VFtoHEDS(vertices, faces)

    # Any mesh processing operation (your final project) using HEDS structure

    # convert back to VF
    newVertices, newFaces = HEDStoVF(verticesArray, halfEdgesArray, facesArray)

    # write the final result
    writeObjFile(newVertices, newFaces, output_name)
    
