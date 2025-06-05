# Computational Geometry final project 2025
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.spatial import Delaunay

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
                face_str = line.strip().split()[1:]
                # Handle potential texture/normal indices by taking only vertex index
                face = [int(v.split('/')[0]) for v in face_str]
                face = [index - 1 for index in face] # Adjust to 0-based indexing
                faces.append(face)
    return vertices, faces

# Half-edge data structure
class Vertex:
    def __init__(self, x, y, z, id=None): 
        self.x = x
        self.y = y
        self.z = z
        self.id = id
        self.halfEdge = None 

    def __repr__(self):
        return f"V(id={self.id}, x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f})"

class Face:
    def __init__(self, id=None): 
        self.id = id
        self.halfEdge = None

    def __repr__(self):
        return f"F(id={self.id})"


class HalfEdge:
    def __init__(self, id=None): 
        self.id = id
        self.origin = None 
        self.twin = None   
        self.face = None   
        self.next = None   
        self.prev = None   

    def __repr__(self):
        orig_id = self.origin.id if self.origin else "None"
        # Destination is origin of next HE
        dest_id = self.next.origin.id if self.next and self.next.origin else "None" 
        face_id = self.face.id if self.face else "None"
        twin_id = self.twin.id if self.twin else "None" # Check twin.id if twin exists
        return f"HE(id={self.id}, O={orig_id}, D={dest_id}, F={face_id}, T={twin_id})"


def VFtoHEDS(vertices_coords, faces_indices):
    """Converts vertex and face lists to Half-Edge Data Structure."""
    verticesArray = {i: Vertex(v[0], v[1], v[2], id=i) for i, v in enumerate(vertices_coords)}
    
    halfedge_id_counter = 0
    halfEdgesArray = [] 
    
    face_id_counter = 0
    facesArray = {} 
    
    edge_to_he_map = {} # (origin_vertex_id, dest_vertex_id) -> halfEdge object

    for face_vertex_indices in faces_indices:
        f = Face(id=face_id_counter)
        facesArray[face_id_counter] = f
        face_id_counter += 1

        face_halfedges_in_loop = []
        num_vertices_in_face = len(face_vertex_indices)

        for i in range(num_vertices_in_face):
            he = HalfEdge(id=halfedge_id_counter)
            halfedge_id_counter += 1
            
            current_vertex_obj_id = face_vertex_indices[i]
            he.origin = verticesArray[current_vertex_obj_id]
            
            # Set vertex.halfEdge if not already set (points to *an* outgoing HE)
            if verticesArray[current_vertex_obj_id].halfEdge is None:
                 verticesArray[current_vertex_obj_id].halfEdge = he
            
            he.face = f
            face_halfedges_in_loop.append(he)
            halfEdgesArray.append(he)

        # Link next/prev for the face loop
        for i in range(num_vertices_in_face):
            he_current = face_halfedges_in_loop[i]
            he_next = face_halfedges_in_loop[(i + 1) % num_vertices_in_face]
            he_prev = face_halfedges_in_loop[(i - 1 + num_vertices_in_face) % num_vertices_in_face]
            
            he_current.next = he_next
            he_current.prev = he_prev
            
            # Add to edge_to_he_map using vertex IDs
            origin_idx = he_current.origin.id
            dest_idx = he_next.origin.id # Destination of he_current is origin of he_next
            edge_to_he_map[(origin_idx, dest_idx)] = he_current

        f.halfEdge = face_halfedges_in_loop[0]

    # Link twins
    for he1 in halfEdgesArray:
        if he1.twin is None: # Process only if not already twinned
            origin_v_id = he1.origin.id
            # Destination vertex of he1 is he1.next.origin
            dest_v_id = he1.next.origin.id
            
            # Find potential twin he2 (dest_v_id -> origin_v_id)
            he2 = edge_to_he_map.get((dest_v_id, origin_v_id))
            if he2:
                he1.twin = he2
                he2.twin = he1
                
    return verticesArray, halfEdgesArray, facesArray, edge_to_he_map


def getFaceVertices(face_obj):
    """Returns a list of Vertex objects for a given Face object."""
    vertices = []
    start_halfEdge = face_obj.halfEdge
    if not start_halfEdge: 
        # This can happen if a face is degenerate or HEDS is malformed
        # print(f"Warning: Face F(id={face_obj.id}) has no halfEdge.")
        return vertices 
    current_halfEdge = start_halfEdge
    visited_hes_in_face = set() # To prevent infinite loops on malformed faces
    while True:
        if current_halfEdge in visited_hes_in_face:
            # print(f"Warning: Loop detected while traversing face F(id={face_obj.id}). Face may be malformed.")
            break
        visited_hes_in_face.add(current_halfEdge)
        
        vertices.append(current_halfEdge.origin)
        current_halfEdge = current_halfEdge.next
        if current_halfEdge == start_halfEdge:
            break
    return vertices

def HEDStoVF(verticesArray, facesArray):
    """Converts HEDS back to vertex and face lists for OBJ export."""
    # Create a stable list of vertices and a map from original id to new index
    ordered_vertices_coords = []
    v_id_to_idx_map = {}
    idx_counter = 0
    # Sort by vertex ID to ensure consistent order if vertices were added/removed
    for v_id in sorted(verticesArray.keys()): 
        v = verticesArray[v_id]
        ordered_vertices_coords.append((v.x, v.y, v.z))
        v_id_to_idx_map[v_id] = idx_counter
        idx_counter +=1

    output_faces_indices = []
    for face_id in sorted(facesArray.keys()): # Sort by face ID
        face_obj = facesArray[face_id]
        face_vertex_objects = getFaceVertices(face_obj)
        if len(face_vertex_objects) < 3: # Skip degenerate faces
            # print(f"Skipping degenerate face F(id={face_id}) with {len(face_vertex_objects)} vertices during HEDStoVF.")
            continue
        # Map vertex objects to their new indices in the ordered_vertices_coords list
        face_indices = [v_id_to_idx_map[v.id] for v in face_vertex_objects]
        output_faces_indices.append(face_indices)
        
    return ordered_vertices_coords, output_faces_indices


def writeObjFile(vertices, faces, output_file):
    """Writes vertex and face lists to an OBJ file."""
    with open(output_file, 'w') as obj_file:
        for vertex_coords in vertices:
            obj_file.write('v ' + ' '.join(map(str, vertex_coords)) + '\n')
        for face_indices in faces:
            # OBJ faces are 1-indexed
            obj_file.write('f ' + ' '.join(map(lambda x: str(x + 1), face_indices)) + '\n')

def visualizeMesh(vertices_coords, faces_indices, holes_v_loops=None, filled_faces_obj_list=None):
    """Visualizes the mesh, detected holes, and filled faces."""
    fig = plt.figure(figsize=(12,10)) # Increased figure size
    ax = fig.add_subplot(111, projection='3d')
    
    if not vertices_coords:
        print("No vertices to visualize.")
        plt.show()
        return
        
    vertices_np_array = np.array(vertices_coords)
    if vertices_np_array.ndim == 1: 
        vertices_np_array = vertices_np_array.reshape(1, -1)

    # Plot vertices
    ax.scatter(vertices_np_array[:, 0], vertices_np_array[:, 1], vertices_np_array[:, 2], c='k', s=10, depthshade=True, label='Vertices')
    
    # Plot edges of all faces (original + filled)
    if faces_indices:
        plotted_original_edges = False
        for face_idx_list in faces_indices:
            if len(face_idx_list) < 3: continue
            face_verts_coords = [vertices_coords[i] for i in face_idx_list]
            face_verts_coords.append(vertices_coords[face_idx_list[0]]) # Close the loop
            face_verts_coords_np = np.array(face_verts_coords)
            if not plotted_original_edges:
                ax.plot(face_verts_coords_np[:, 0], face_verts_coords_np[:, 1], face_verts_coords_np[:, 2], c='b', alpha=0.3, label='Original/Existing Faces Edges')
                plotted_original_edges = True
            else:
                ax.plot(face_verts_coords_np[:, 0], face_verts_coords_np[:, 1], face_verts_coords_np[:, 2], c='b', alpha=0.3)


    # Plot detected hole boundaries
    if holes_v_loops:
        plotted_hole_boundaries = False
        for hole_v_loop in holes_v_loops: 
            if not hole_v_loop or len(hole_v_loop) < 2 : continue
            x = [v.x for v in hole_v_loop] + [hole_v_loop[0].x]
            y = [v.y for v in hole_v_loop] + [hole_v_loop[0].y]
            z = [v.z for v in hole_v_loop] + [hole_v_loop[0].z]
            if not plotted_hole_boundaries:
                ax.plot(x, y, z, color='r', linewidth=3, label='Detected Hole Boundaries')
                plotted_hole_boundaries = True
            else:
                 ax.plot(x, y, z, color='r', linewidth=3)

    # Plot filled faces (edges of new triangles)
    if filled_faces_obj_list:
        plotted_filled_edges = False
        for face_obj in filled_faces_obj_list: 
            verts_obj_list = getFaceVertices(face_obj)
            if not verts_obj_list or len(verts_obj_list) < 3 : continue
            x = [v.x for v in verts_obj_list] + [verts_obj_list[0].x]
            y = [v.y for v in verts_obj_list] + [verts_obj_list[0].y]
            z = [v.z for v in verts_obj_list] + [verts_obj_list[0].z]
            if not plotted_filled_edges:
                ax.plot(x, y, z, color='g', linewidth=1.5, alpha=0.7, label='Filled Faces Edges')
                plotted_filled_edges = True
            else:
                ax.plot(x, y, z, color='g', linewidth=1.5, alpha=0.7)

    # Set plot limits and labels
    if vertices_np_array.size > 0 :
        all_x, all_y, all_z = vertices_np_array[:,0], vertices_np_array[:,1], vertices_np_array[:,2]
        
        max_range = np.array([all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()]).max() 
        if max_range == 0: max_range = 1.0 # Avoid division by zero if all points are the same
        
        mid_x, mid_y, mid_z = (all_x.max()+all_x.min())*0.5, (all_y.max()+all_y.min())*0.5, (all_z.max()+all_z.min())*0.5
        
        ax.set_xlim(mid_x - max_range*0.6, mid_x + max_range*0.6)
        ax.set_ylim(mid_y - max_range*0.6, mid_y + max_range*0.6)
        ax.set_zlim(mid_z - max_range*0.6, mid_z + max_range*0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z') 
    ax.legend()
    plt.title("Mesh Visualization")
    plt.show()


def detect_holes(halfEdgesArray, ignore_largest=False):
    """Detects hole boundaries (loops of Vertex objects)."""
    visited_he_for_hole_finding = set() 
    all_loops_vertex_lists = [] 

    for he_start_candidate in halfEdgesArray:
        # A boundary half-edge has no twin
        if he_start_candidate.twin is None and he_start_candidate not in visited_he_for_hole_finding:
            current_loop_vertices = []
            current_he_in_loop = he_start_candidate
            
            # Path to trace the current hole boundary
            path_taken_hes_in_current_loop = set() 

            while current_he_in_loop not in path_taken_hes_in_current_loop : 
                path_taken_hes_in_current_loop.add(current_he_in_loop)
                # Mark the starting HE of this loop segment as visited for overall hole finding
                visited_he_for_hole_finding.add(current_he_in_loop) 
                current_loop_vertices.append(current_he_in_loop.origin)
                
                # Traverse to the next HE that forms the boundary.
                # This is effectively "turning left" at each vertex along the boundary.
                # Start by trying to go to the next HE in the current (non-existent) face.
                next_he_on_boundary_candidate = current_he_in_loop.next 
                # Then, keep pivoting around the destination vertex of current_he_in_loop
                # (which is next_he_on_boundary_candidate.origin) by taking twin.next
                # until we find an edge that has no twin (is a boundary edge).
                while next_he_on_boundary_candidate.twin is not None:
                    next_he_on_boundary_candidate = next_he_on_boundary_candidate.twin.next
                    # Safety break for complex configurations or errors
                    if next_he_on_boundary_candidate == current_he_in_loop.next: 
                        # print("Warning: Hole detection might have issues with complex vertex configurations or non-manifold geometry.")
                        current_loop_vertices = [] # Discard this loop attempt
                        break
                
                if not current_loop_vertices: # If loop was discarded
                    break

                current_he_in_loop = next_he_on_boundary_candidate
                if current_he_in_loop == he_start_candidate: # Closed the loop
                    break
                # If we somehow land on an edge that is not a boundary edge, something is wrong.
                if current_he_in_loop.twin is not None and current_he_in_loop != he_start_candidate :
                    # print(f"Warning: Hole detection strayed from boundary at HE {current_he_in_loop.id}. Discarding loop.")
                    current_loop_vertices = [] 
                    break
            
            if current_loop_vertices: 
                all_loops_vertex_lists.append(current_loop_vertices)
                
    print(f"{len(all_loops_vertex_lists)} hole boundary loop(s) detected initially.")

    if ignore_largest and len(all_loops_vertex_lists) > 1:
        all_loops_vertex_lists.sort(key=len, reverse=True) # Sort by num vertices, descending
        return all_loops_vertex_lists[1:] # Return all but the largest
    return all_loops_vertex_lists


def _add_new_face_to_heds(v_obj_list, verticesArray, halfEdgesArray, facesArray, edge_to_he_map, he_id_gen, face_id_gen):
    """ Helper to create one new triangular face and update HEDS.
        v_obj_list is [v0, v1, v2] (Vertex objects) assumed to be in CCW order for the new face.
    """
    v0, v1, v2 = v_obj_list[0], v_obj_list[1], v_obj_list[2]

    new_face_id = next(face_id_gen)
    new_face = Face(id=new_face_id)
    facesArray[new_face_id] = new_face
    
    he0_id, he1_id, he2_id = next(he_id_gen), next(he_id_gen), next(he_id_gen)
    he0 = HalfEdge(id=he0_id); he1 = HalfEdge(id=he1_id); he2 = HalfEdge(id=he2_id)
    
    halfEdgesArray.extend([he0, he1, he2])

    # Set origins
    he0.origin = v0; he1.origin = v1; he2.origin = v2

    # Set halfEdge pointer for vertices if not already set (especially for new vertices)
    if v0.halfEdge is None: v0.halfEdge = he0
    if v1.halfEdge is None: v1.halfEdge = he1
    if v2.halfEdge is None: v2.halfEdge = he2
    
    # Set face
    he0.face = new_face; he1.face = new_face; he2.face = new_face
    new_face.halfEdge = he0 # Point face to one of its half-edges

    # Set next/prev
    he0.next = he1; he1.next = he2; he2.next = he0
    he0.prev = he2; he1.prev = he0; he2.prev = he1

    # Update edge_to_he_map with new half-edges (using vertex IDs)
    edge_to_he_map[(v0.id, v1.id)] = he0
    edge_to_he_map[(v1.id, v2.id)] = he1
    edge_to_he_map[(v2.id, v0.id)] = he2
    
    # Link twins for the three new half-edges
    edges_to_link = [(v0,v1,he0), (v1,v2,he1), (v2,v0,he2)]
    for orig_v, dest_v, current_he in edges_to_link:
        # Look for the reverse edge (dest_v.id -> orig_v.id) in the map
        twin_he = edge_to_he_map.get((dest_v.id, orig_v.id))
        if twin_he:
            current_he.twin = twin_he
            # If the found twin was a boundary edge (twin_he.twin was None), update it.
            # Or if it was an internal edge of another part of the patch, it should also be None initially.
            if twin_he.twin is None : 
                twin_he.twin = current_he
            # else:
                # This case (twin_he.twin is not None) implies we are overwriting an existing twin.
                # This can happen if edge_to_he_map had an older HE for (dest_v.id, orig_v.id)
                # that was already part of another new face in this patch.
                # The map should always point to the "latest" HE for a directed edge.
                # print(f"Warning: HE {twin_he.id} (O:{twin_he.origin.id} D:{twin_he.next.origin.id}) already had a twin {twin_he.twin.id}. Overwriting with {current_he.id}")
                # twin_he.twin = current_he # Ensure it's set
        
    return new_face


def get_pca_plane_normal(points_3d_np):
    """Calculates the normal of the best-fit plane for a set of 3D points using PCA."""
    if points_3d_np.shape[0] < 3 or np.linalg.matrix_rank(points_3d_np - np.mean(points_3d_np, axis=0)) < 2:
        # print("Warning: Not enough distinct points to define a plane robustly via PCA. Using default Z-up normal.")
        return np.array([0,0,1]), np.mean(points_3d_np, axis=0), None, None # Return default normal, centroid, and None for axes
        
    centroid = np.mean(points_3d_np, axis=0)
    centered_pts = points_3d_np - centroid
    cov_matrix = np.cov(centered_pts.T)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix) # eigh for symmetric, sorts eigenvalues
    
    plane_normal = eigen_vectors[:, 0] # Normal is eigenvector for smallest eigenvalue
    u_axis = eigen_vectors[:, 2]       # PCA major axis in plane
    v_axis = eigen_vectors[:, 1]       # PCA minor axis in plane
    return plane_normal, centroid, u_axis, v_axis


def fill_triangular_hole(hole_v_loop, verticesArray, halfEdgesArray, facesArray, edge_to_he_map, he_id_gen, face_id_gen):
    """Fills a 3-vertex hole, ensuring correct normal orientation."""
    # hole_v_loop is [v0, v1, v2] (Vertex objects)
    # print(f"Attempting to fill triangular hole with vertices: {[v.id for v in hole_v_loop]}")
    v0, v1, v2 = hole_v_loop[0], hole_v_loop[1], hole_v_loop[2]
    
    p0 = np.array([v0.x, v0.y, v0.z])
    p1 = np.array([v1.x, v1.y, v1.z])
    p2 = np.array([v2.x, v2.y, v2.z])
    hole_points_3d = np.array([p0,p1,p2])

    expected_plane_normal, _, _, _ = get_pca_plane_normal(hole_points_3d)
    if expected_plane_normal is None: # Fallback if PCA failed (e.g. collinear)
        print(f"Warning: Degenerate (collinear) 3-vertex hole: {[v.id for v in hole_v_loop]}. Skipping fill.")
        return []

    current_tri_normal = np.cross(p1 - p0, p2 - p0)
    norm_val = np.linalg.norm(current_tri_normal)
    if norm_val < 1e-9: # Degenerate triangle (collinear points)
        print(f"Warning: Degenerate triangle from hole loop {[v.id for v in hole_v_loop]}. Skipping fill.")
        return []
    current_tri_normal /= norm_val

    final_v_list = [v0, v1, v2]
    if np.dot(current_tri_normal, expected_plane_normal) < 0:
        final_v_list = [v0, v2, v1] # Flip to align with expected normal
        
    new_face = _add_new_face_to_heds(final_v_list, verticesArray, halfEdgesArray, facesArray, edge_to_he_map, he_id_gen, face_id_gen)
    return [new_face]


def fill_hole_fan_triangulation(hole_v_loop, verticesArray, halfEdgesArray, facesArray, edge_to_he_map, he_id_gen, face_id_gen, vertex_id_gen):
    """Fills a hole (e.g., 4-6 vertices) using fan triangulation from a new central vertex."""
    num_hole_verts = len(hole_v_loop)
    # print(f"Attempting to fill {num_hole_verts}-gon hole (fan) with vertices: {[v.id for v in hole_v_loop]}")
    
    hole_points_3d_np = np.array([[v.x, v.y, v.z] for v in hole_v_loop])
    pca_plane_normal, _, _, _ = get_pca_plane_normal(hole_points_3d_np)
    if pca_plane_normal is None: # PCA failed
        print(f"Warning: Cannot determine plane normal for fan triangulation of hole {[v.id for v in hole_v_loop]}. Using default Z-up normal for orientation.")
        pca_plane_normal = np.array([0,0,1]) 

    # 1. Create new central vertex
    centroid_coords = np.mean(hole_points_3d_np, axis=0)
    vc_id = next(vertex_id_gen)
    vc = Vertex(centroid_coords[0], centroid_coords[1], centroid_coords[2], id=vc_id)
    verticesArray[vc_id] = vc

    filled_faces_list = []
    # 2. Create N triangular faces (vi, v_next, vc)
    for i in range(num_hole_verts):
        vi = hole_v_loop[i]
        v_next = hole_v_loop[(i + 1) % num_hole_verts] # Ensures loop closes
        
        p_vi = np.array([vi.x, vi.y, vi.z])
        p_vnext = np.array([v_next.x, v_next.y, v_next.z])
        p_vc = np.array([vc.x, vc.y, vc.z])
        
        # Normal of triangle (vi, v_next, vc)
        current_tri_normal = np.cross(p_vnext - p_vi, p_vc - p_vi)
        norm_val = np.linalg.norm(current_tri_normal)
        if norm_val < 1e-9: continue # Skip degenerate triangle
        current_tri_normal /= norm_val
        
        current_v_list = [vi, v_next, vc]
        if np.dot(current_tri_normal, pca_plane_normal) < 0:
            current_v_list = [vi, vc, v_next] # Flip order to align normal
            
        new_face = _add_new_face_to_heds(current_v_list, verticesArray, halfEdgesArray, facesArray, edge_to_he_map, he_id_gen, face_id_gen)
        filled_faces_list.append(new_face)
        
    return filled_faces_list


def fill_hole_delaunay_triangulation(hole_v_loop, verticesArray, halfEdgesArray, facesArray, edge_to_he_map, he_id_gen, face_id_gen, vertex_id_gen):
    """Fills a larger hole using PCA projection and Delaunay triangulation, with filtering."""
    num_hole_verts = len(hole_v_loop)
    # print(f"Attempting to fill {num_hole_verts}-gon hole (Delaunay) with vertices: {[v.id for v in hole_v_loop]}")

    hole_points_3d = np.array([[v.x, v.y, v.z] for v in hole_v_loop])

    # 1. PCA to get plane normal, centroid, and projection axes
    plane_normal, centroid_3d, u_axis, v_axis = get_pca_plane_normal(hole_points_3d)
    if plane_normal is None or u_axis is None or v_axis is None: # PCA failed
        print(f"Warning: PCA failed for Delaunay hole {[v.id for v in hole_v_loop]}. Attempting fallback fan triangulation.")
        # Fallback to fan triangulation if PCA is not possible
        if num_hole_verts > 0 :
             return fill_hole_fan_triangulation(hole_v_loop, verticesArray, halfEdgesArray, facesArray, edge_to_he_map, he_id_gen, face_id_gen, vertex_id_gen)
        return []


    centered_points_3d = hole_points_3d - centroid_3d
    proj_2d_boundary_pts = np.array([[np.dot(p, u_axis), np.dot(p, v_axis)] for p in centered_points_3d])
    
    # This is the critical path for filtering Delaunay triangles
    original_hole_2d_path_for_filtering = Path(proj_2d_boundary_pts) 

    points_for_delaunay_2d = list(proj_2d_boundary_pts) # Start with boundary points

    # Optional: Add internal points for balanced triangulation
    if num_hole_verts > 6: # Only for larger holes
        min_coords = np.min(proj_2d_boundary_pts, axis=0)
        max_coords = np.max(proj_2d_boundary_pts, axis=0)
        
        # Calculate average edge length on boundary for spacing heuristic
        edge_lengths_2d = [np.linalg.norm(proj_2d_boundary_pts[i] - proj_2d_boundary_pts[(i + 1) % num_hole_verts]) for i in range(num_hole_verts)]
        spacing = np.mean(edge_lengths_2d) * 0.75 

        if spacing > 1e-4: # Avoid too small spacing
            x_vals = np.arange(min_coords[0] + spacing*0.1, max_coords[0] - spacing*0.1, spacing) 
            y_vals = np.arange(min_coords[1] + spacing*0.1, max_coords[1] - spacing*0.1, spacing)
            if len(x_vals)>0 and len(y_vals)>0:
                xx, yy = np.meshgrid(x_vals, y_vals)
                grid_points_2d = np.vstack([xx.ravel(), yy.ravel()]).T
                
                if grid_points_2d.size > 0:
                    # Filter grid points to be inside the 2D hole polygon
                    inside_grid_points = grid_points_2d[original_hole_2d_path_for_filtering.contains_points(grid_points_2d, radius=-spacing*0.05)] # Negative radius to shrink slightly
                    points_for_delaunay_2d.extend(inside_grid_points)
    
    points_for_delaunay_np_2d = np.array(points_for_delaunay_2d)
    if points_for_delaunay_np_2d.shape[0] < 3:
        # print(f"Warning: Not enough points ({points_for_delaunay_np_2d.shape[0]}) for Delaunay for hole {[v.id for v in hole_v_loop]}. Attempting fallback.")
        if num_hole_verts > 0:
            return fill_hole_fan_triangulation(hole_v_loop, verticesArray, halfEdgesArray, facesArray, edge_to_he_map, he_id_gen, face_id_gen, vertex_id_gen)
        return []

    try:
        triangulation_2d = Delaunay(points_for_delaunay_np_2d)
    except Exception as e:
        # print(f"Error during Delaunay for hole {[v.id for v in hole_v_loop]}: {e}. Attempting fallback.")
        if num_hole_verts > 0:
            return fill_hole_fan_triangulation(hole_v_loop, verticesArray, halfEdgesArray, facesArray, edge_to_he_map, he_id_gen, face_id_gen, vertex_id_gen)
        return []

    # plot_delaunay_2d(points_for_delaunay_np_2d, triangulation_2d, proj_2d_boundary_pts) # For debugging

    filled_faces_list = []
    # Map 2D Delaunay point indices back to 3D Vertex objects
    delaunay_idx_to_vertex_obj_map = {} 

    # Original boundary vertices (first num_hole_verts in points_for_delaunay_np_2d)
    for i in range(num_hole_verts):
        delaunay_idx_to_vertex_obj_map[i] = hole_v_loop[i]
    
    # New internal vertices (the rest of points_for_delaunay_np_2d)
    for i in range(num_hole_verts, len(points_for_delaunay_np_2d)):
        pt_2d = points_for_delaunay_np_2d[i]
        # Unproject: pt_3d = centroid_3d + pt_2d[0]*u_axis + pt_2d[1]*v_axis
        pt_3d_coords = centroid_3d + pt_2d[0] * u_axis + pt_2d[1] * v_axis
        
        new_v_id = next(vertex_id_gen)
        new_v_obj = Vertex(pt_3d_coords[0], pt_3d_coords[1], pt_3d_coords[2], id=new_v_id)
        verticesArray[new_v_id] = new_v_obj
        delaunay_idx_to_vertex_obj_map[i] = new_v_obj

    # Create 3D faces from filtered 2D Delaunay triangles
    for simplex_indices in triangulation_2d.simplices: 
        idx0, idx1, idx2 = simplex_indices
        
        # Get 2D points of the simplex for centroid calculation
        pt0_2d = points_for_delaunay_np_2d[idx0]
        pt1_2d = points_for_delaunay_np_2d[idx1]
        pt2_2d = points_for_delaunay_np_2d[idx2]
        simplex_centroid_2d = (pt0_2d + pt1_2d + pt2_2d) / 3.0

        # *** CRITICAL FILTERING STEP ***
        if not original_hole_2d_path_for_filtering.contains_point(simplex_centroid_2d, radius=1e-5): # Add small radius for robustness
            # print(f"Skipping Delaunay triangle outside original hole boundary. Centroid: {simplex_centroid_2d}")
            continue 

        v_objects_for_face = [delaunay_idx_to_vertex_obj_map[idx0],
                              delaunay_idx_to_vertex_obj_map[idx1],
                              delaunay_idx_to_vertex_obj_map[idx2]]
        
        p0_3d = np.array([v_objects_for_face[0].x, v_objects_for_face[0].y, v_objects_for_face[0].z])
        p1_3d = np.array([v_objects_for_face[1].x, v_objects_for_face[1].y, v_objects_for_face[1].z])
        p2_3d = np.array([v_objects_for_face[2].x, v_objects_for_face[2].y, v_objects_for_face[2].z])
        
        current_tri_normal_3d = np.cross(p1_3d - p0_3d, p2_3d - p0_3d)
        norm_val = np.linalg.norm(current_tri_normal_3d)
        if norm_val < 1e-9: continue # Skip degenerate 3D triangle
        current_tri_normal_3d /= norm_val

        final_v_list_for_face = list(v_objects_for_face) # Make a mutable copy
        if np.dot(current_tri_normal_3d, plane_normal) < 0: # Align with overall hole plane normal
            final_v_list_for_face = [v_objects_for_face[0], v_objects_for_face[2], v_objects_for_face[1]] 

        new_face = _add_new_face_to_heds(final_v_list_for_face, verticesArray, halfEdgesArray, facesArray, edge_to_he_map, he_id_gen, face_id_gen)
        filled_faces_list.append(new_face)
        
    return filled_faces_list


def plot_delaunay_2d(points_2d_all, delaunay_obj, boundary_points_2d=None): 
    """Plots the 2D Delaunay triangulation for debugging."""
    plt.figure(figsize=(8, 8))
    plt.triplot(points_2d_all[:, 0], points_2d_all[:, 1], delaunay_obj.simplices, color='gray', lw=0.5)
    plt.plot(points_2d_all[:, 0], points_2d_all[:, 1], 'o', color='blue', markersize=3, label='All points for Delaunay')
    if boundary_points_2d is not None and boundary_points_2d.shape[0] > 0:
        # Close the boundary loop for plotting
        closed_boundary = np.vstack([boundary_points_2d, boundary_points_2d[0]])
        plt.plot(closed_boundary[:, 0], closed_boundary[:, 1], 'r-', lw=1.5, label='Original Hole Boundary (2D Projection)')
    plt.gca().set_aspect('equal')
    plt.title('2D Delaunay Triangulation (Projected Hole Plane)')
    plt.xlabel('u (PCA axis 1)')
    plt.ylabel('v (PCA axis 2)')
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- ID Generators ---
def id_generator(start_val=0):
    """Simple generator for unique IDs."""
    count = start_val
    while True:
        yield count
        count += 1

# --- Main Processing Function ---
def process_mesh(input_name: str, output_name: str, ignore_outer_boundary_param=True, fan_threshold=6):
    vertices_coords, faces_indices = readObjFile(input_name)
    if not vertices_coords:
        print(f"No vertices found in {input_name}. Exiting.")
        return

    print(f"Read {len(vertices_coords)} vertices and {len(faces_indices)} faces from {input_name}")

    verticesArray, halfEdgesArray, facesArray, edge_to_he_map = VFtoHEDS(vertices_coords, faces_indices)
    
    # Initialize ID generators based on current max IDs + 1
    max_v_id = max(verticesArray.keys()) if verticesArray else -1
    max_f_id = max(facesArray.keys()) if facesArray else -1
    max_he_id = max((he.id for he in halfEdgesArray if he.id is not None), default=-1) if halfEdgesArray else -1

    vertex_id_gen = id_generator(max_v_id + 1)
    face_id_gen = id_generator(max_f_id + 1)
    he_id_gen = id_generator(max_he_id + 1)
    
    # Detect holes (returns list of lists of Vertex objects)
    # The parameter to detect_holes is 'ignore_largest'
    hole_vertex_loops_for_filling = detect_holes(halfEdgesArray, ignore_largest=ignore_outer_boundary_param)
    
    # For visualization, we might want to show ALL detected loops before filtering
    all_detected_hole_loops_for_viz = detect_holes(halfEdgesArray, ignore_largest=False)


    all_newly_filled_face_objects = [] 

    if not hole_vertex_loops_for_filling:
        print("No holes targeted for filling (after considering 'ignore_outer_boundary').")
    else:
        print(f"Processing {len(hole_vertex_loops_for_filling)} hole(s) for filling...")
        for i, hole_v_loop in enumerate(hole_vertex_loops_for_filling):
            num_hole_verts = len(hole_v_loop)
            # print(f"  Hole {i+1}/{len(hole_vertex_loops_for_filling)}: {num_hole_verts} vertices - IDs {[v.id for v in hole_v_loop[:5]]}...")

            if num_hole_verts < 3:
                # print(f"  Skipping degenerate hole with {num_hole_verts} vertices.")
                continue

            created_faces_for_this_hole = []
            if num_hole_verts == 3:
                created_faces_for_this_hole = fill_triangular_hole(
                    hole_v_loop, verticesArray, halfEdgesArray, facesArray, edge_to_he_map, he_id_gen, face_id_gen
                )
            elif num_hole_verts <= fan_threshold: 
                created_faces_for_this_hole = fill_hole_fan_triangulation(
                    hole_v_loop, verticesArray, halfEdgesArray, facesArray, edge_to_he_map, he_id_gen, face_id_gen, vertex_id_gen
                )
            else: # Larger holes use Delaunay
                 created_faces_for_this_hole = fill_hole_delaunay_triangulation(
                    hole_v_loop, verticesArray, halfEdgesArray, facesArray, edge_to_he_map, he_id_gen, face_id_gen, vertex_id_gen
                )
            all_newly_filled_face_objects.extend(created_faces_for_this_hole)
    
    print(f"Total new faces created: {len(all_newly_filled_face_objects)}")
    
    # Convert HEDS back to V,F for writing and visualization
    final_vertices_coords, final_faces_indices = HEDStoVF(verticesArray, facesArray)
    
    print(f"Writing processed mesh ({len(final_vertices_coords)} V, {len(final_faces_indices)} F) to {output_name}")
    writeObjFile(final_vertices_coords, final_faces_indices, output_name)
    
    print("Visualizing mesh...")
    # Pass the loops that were actually targeted for filling, or all for full context
    visualizeMesh(final_vertices_coords, final_faces_indices, 
                  holes_v_loops=all_detected_hole_loops_for_viz, # Show all detected boundaries in red
                  filled_faces_obj_list=all_newly_filled_face_objects) # Show new green faces
