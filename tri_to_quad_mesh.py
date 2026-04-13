import numpy as np
import torch
import trimesh
from pathlib import Path


def load_obj_with_uv(filename):
    """
    Loads an OBJ file and extracts:
    - 3D vertex positions (V)
    - triangle faces (F)
    - UV coordinates (uv)
    """
    mesh = trimesh.load(filename, process=False)
    V = mesh.vertices
    F = mesh.faces
    # UV coordinates
    if hasattr(mesh.visual, 'uv'):
        uv = mesh.visual.uv
    else:
        raise ValueError("No UV coordinates found in OBJ file.")

    return np.array(V), np.array(F), np.array(uv)


def generate_uv_grid(nx, ny, inv_u=False, inv_v=False, flatten=True):
    """
    Generate a UV grid with configurable directions.

    Parameters:
    - nx, ny: number of grid points in u and v
    - udir: 'lr' (left to right) or 'rl' (right to left)
    - vdir: 'tb' (top to bottom) or 'bt' (bottom to top)
    - flatten: if True, return (N, 2) flattened UV grid; else (ny, nx, 2)

    Returns:
    - grid_uv: np.ndarray of shape (N, 2) or (ny, nx, 2)
    """
    u = np.linspace(0, 1, nx) if not inv_u else np.linspace(1, 0, nx)
    v = np.linspace(0, 1, ny) if not inv_v else np.linspace(1, 0, ny)

    uu, vv = np.meshgrid(u, v)  # shape (ny, nx)
    uv_grid = np.stack([uu, vv], axis=-1)

    if flatten:
        return uv_grid.reshape(-1, 2)
    return uv_grid  # shape: (ny, nx, 2)



def barycentric_interp(uv_tri, v_tri, p):
    """
    Interpolates 3D position at point p in UV space using barycentric coordinates
    uv_tri: (3, 2) triangle in UV
    v_tri: (3, 3) triangle in 3D
    p: (2,) query point
    Returns interpolated (3,) position
    """
    A = np.array([
        [uv_tri[0, 0] - uv_tri[2, 0], uv_tri[1, 0] - uv_tri[2, 0]],
        [uv_tri[0, 1] - uv_tri[2, 1], uv_tri[1, 1] - uv_tri[2, 1]]
    ])
    b = p - uv_tri[2]
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return v_tri[2]  # fallback if degenerate
    w0, w1 = w
    w2 = 1 - w0 - w1
    return w0 * v_tri[0] + w1 * v_tri[1] + w2 * v_tri[2]


def uv_to_3d_positions(grid_uv, uv, V, F):
    """
    For each UV grid point, find which triangle it lies in and interpolate its 3D position.
    """
    from matplotlib.tri import Triangulation

    tri = Triangulation(uv[:, 0], uv[:, 1], F)
    interpolated_positions = []

    for point in grid_uv:
        simplex = tri.get_trifinder()(point[0], point[1])
        if simplex == -1:
            interpolated_positions.append([np.nan, np.nan, np.nan])
            continue
        tri_inds = F[simplex]
        uv_tri = uv[tri_inds]
        v_tri = V[tri_inds]
        pos = barycentric_interp(uv_tri, v_tri, point)
        interpolated_positions.append(pos)

    return np.array(interpolated_positions)


def trimesh_to_quadmesh_from_uv(obj_path, resolution=(50, 50)):
    """
    Main function to load a mesh with UV, generate uniform UV grid, and map to 3D quad mesh.
    Returns:
        - vertices_3d: (Nx, Ny, 3) grid of 3D points
        - faces_quad: (M, 4) list of quad face indices (optional for reconstruction)
    """
    V, F, uv = load_obj_with_uv(obj_path)
    nx, ny = resolution
    # grid_uv = generate_uniform_grid_uv(nx, ny)
    grid_uv = generate_uv_grid(nx, ny, inv_u=False, inv_v=True, flatten=True)
    interpolated_positions = uv_to_3d_positions(grid_uv, uv, V, F)

    # Reshape to (nx, ny, 3) grid
    vertices_3d = interpolated_positions.reshape((ny, nx, 3))

    # Optionally generate quad faces (as 2x triangles or quad face indices)
    quads = []
    for j in range(nx - 1):  # column
        for i in range(ny - 1):  # row
            idx0 = j * ny + i
            idx1 = (j + 1) * ny + i
            idx2 = (j + 1) * ny + i + 1
            idx3 = j * ny + i + 1
            quads.append([idx0, idx1, idx2, idx3])
    return vertices_3d, np.array(quads)


def save_quadmesh_obj_with_uv(filename, vertices_3d, faces_quad, uv_grid):
    """
    Save quad mesh with custom-ordered UVs and faces in OBJ format.
    - vertices_3d: (ny, nx, 3)
    - faces_quad: (M, 4)
    - uv_grid: (ny, nx, 2)
    """
    ny, nx, _ = vertices_3d.shape
    verts_flat = vertices_3d.reshape(-1, 3)

    # Reorder UVs: column-wise (for u), top to bottom (for v decreasing)
    uv_lines = []
    uv_indices = np.zeros((ny, nx), dtype=int)
    counter = 1
    for j in range(nx):  # left to right (u)
        for i in range(ny):  # top to bottom (v descending)
            uv = uv_grid[i, j]  # (v goes from top to bottom)
            uv_lines.append(f"vt {uv[0]} {uv[1]}")
            uv_indices[i, j] = counter
            counter += 1

    # Now write obj
    with open(filename, 'w') as f:
        # Write vertices
        for v in verts_flat:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        # Write UVs
        for uv_line in uv_lines:
            f.write(f"{uv_line}\n")

        # Write faces: vertex/uv index
        for face in faces_quad:
            v0, v1, v2, v3 = face  # vertex indices
            f.write(f"f {v0 + 1}/{v0 + 1} {v1 + 1}/{v1 + 1} {v2 + 1}/{v2 + 1} {v3 + 1}/{v3 + 1}\n")
    print(f"Saved quad mesh with UVs to {filename}")


def batch_trimesh_to_quadmesh(vertices_seq, faces, uv, resolution=(50, 50)):
    """
    Args:
        vertices_seq: (N_frames, n_vertices, 3) array
        faces: (n_faces, 3) triangle indices
        uv: (n_faces, 2) UV coords for each triangle corner
        resolution: UV grid resolution (nx, ny)
    Returns:
        quad_verts_tensor: (N_frames, ny, nx, 3)
        quads: (n_quads, 3) index tensor for quad faces
    """
    nx, ny = resolution
    grid_uv = generate_uv_grid(nx, ny, inv_u=False, inv_v=True, flatten=True)  # (nx*ny, 2)

    N_frames = vertices_seq.shape[0]
    interpolated_list = []
    for t in range(N_frames):
        V_t = vertices_seq[t]  # (n_vertices, 3)
        interp = uv_to_3d_positions(grid_uv, uv, V_t, faces)  # (nx*ny, 3)
        interpolated_list.append(interp.reshape(ny, nx, 3))

    quad_verts_tensor = np.stack(interpolated_list, axis=0)  # (N_frames, ny, nx, 3)

    # Generate quad faces once (same as before)
    quads = []
    for j in range(nx - 1):  # column
        for i in range(ny - 1):  # row
            idx0 = j * ny + i
            idx1 = (j + 1) * ny + i
            idx2 = (j + 1) * ny + i + 1
            idx3 = j * ny + i + 1
            quads.append([idx0, idx1, idx2, idx3])
    quads = np.array(quads)

    return quad_verts_tensor, quads



def batch_trimesh_to_quadmesh_torch(vertices_seq, faces, uv_per_vertex, resolution=(50, 50), eps=1e-6):
    device = vertices_seq.device
    N, V, _ = vertices_seq.shape
    nx, ny = resolution

    grid_uv_np = generate_uv_grid(nx, ny, inv_u=False, inv_v=True, flatten=True)  # (nx*ny,2)
    grid_uv = torch.from_numpy(grid_uv_np).to(device).double()
    M = grid_uv.shape[0]

    uv = uv_per_vertex.to(device).double()  # (V,2)
    F = faces.to(device)  # (F,3)
    uv_f = uv[F]  # (F,3,2)
    p0, p1, p2 = uv_f.unbind(dim=1)  # each (F,2)
    v0 = p1 - p0  # (F,2)
    v1 = p2 - p0  # (F,2)

    dot00 = (v0 * v0).sum(1)  # (F,)
    dot01 = (v0 * v1).sum(1)
    dot11 = (v1 * v1).sum(1)
    invDen = 1.0 / (dot00 * dot11 - dot01 * dot01)  # (F,)

    guv = grid_uv.unsqueeze(1)  # (M,1,2)
    p0e = p0.unsqueeze(0)  # (1,F,2)
    v0e = v0.unsqueeze(0)
    v1e = v1.unsqueeze(0)
    invDe = invDen.unsqueeze(0)  # (1,F)

    v2 = guv - p0e  # (M,F,2)
    dot02 = (v0e * v2).sum(2)  # (M,F)
    dot12 = (v1e * v2).sum(2)  # (M,F)

    u_coords = (dot11.unsqueeze(0) * dot02 - dot01.unsqueeze(0) * dot12) * invDe  # (M,F)
    v_coords = (dot00.unsqueeze(0) * dot12 - dot01.unsqueeze(0) * dot02) * invDe  # (M,F)

    cond = (u_coords >= -eps) & (v_coords >= -eps) & (u_coords + v_coords <= 1 + eps)  # (M,F)
    face_idx = cond.float().argmax(1)  # (M,)

    u_sel = u_coords[torch.arange(M, device=device), face_idx]
    v_sel = v_coords[torch.arange(M, device=device), face_idx]
    w_sel = 1.0 - u_sel - v_sel
    u_sel = u_sel.clamp(0.0, 1.0)
    v_sel = v_sel.clamp(0.0, 1.0)
    w_sel = w_sel.clamp(0.0, 1.0)

    f0 = F[face_idx, 0]
    f1 = F[face_idx, 1]
    f2 = F[face_idx, 2]

    v0_3d = vertices_seq[:, f0, :].double()
    v1_3d = vertices_seq[:, f1, :].double()
    v2_3d = vertices_seq[:, f2, :].double()

    w = w_sel.view(1, M, 1)
    u_w = u_sel.view(1, M, 1)
    v_w = v_sel.view(1, M, 1)
    pts = w * v0_3d + u_w * v1_3d + v_w * v2_3d  # (N, M, 3)

    quad_verts = pts.view(N, ny, nx, 3).float()

    quads = []
    for j in range(nx - 1):
        for i in range(ny - 1):
            idx0 = j * ny + i
            idx1 = (j + 1) * ny + i
            idx2 = (j + 1) * ny + (i + 1)
            idx3 = j * ny + (i + 1)
            quads.append([idx0, idx1, idx2, idx3])
    quads = torch.tensor(quads, dtype=torch.long, device=device)

    return quad_verts, quads


if __name__ == "__main__":
    obj_example_path = "./square_1024.obj"
    resolution = 16
    quad_verts, quad_faces = trimesh_to_quadmesh_from_uv(obj_example_path, resolution=(resolution, resolution))

    # debug
    V, F, uv = load_obj_with_uv(obj_example_path)
    V = np.array(V)
    F = np.array(F)
    uv = np.array(uv)
    quad_verts2, quads2 = batch_trimesh_to_quadmesh(V[None], F, uv, (resolution, resolution))

    # save the quad mesh to a new OBJ file
    output_path = Path(obj_example_path).parent / f"square_quad_res{resolution}.obj"
    # uv_grid = generate_regular_uv_grid(resolution, resolution)
    uv_grid = generate_uv_grid(resolution, resolution, inv_u=True, inv_v=True, flatten=False)
    save_quadmesh_obj_with_uv(output_path, quad_verts, quad_faces, uv_grid)

    print('Saved quad mesh to:', output_path)