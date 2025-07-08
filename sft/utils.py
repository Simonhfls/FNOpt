import json
import openmesh as om
import torch
from matplotlib import pyplot as plt


def save_mesh(v, f, path):
    mesh = om.TriMesh()

    vertex_handles = []
    for vertex in v:
        vh = mesh.add_vertex(vertex)
        vertex_handles.append(vh)

    for face in f:
        face_vertices = [vertex_handles[vi] for vi in face]
        mesh.add_face(face_vertices)

    om.write_mesh(path, mesh)
    print(f'Mesh saved to {path}')

def save_pcl(pcl, filepath):
    with open(filepath, 'w') as f:
        for line in pcl:
            if len(line) == 3:
                f.write(f"v {line[0]} {line[1]} {line[2]}\n")
            else:
                f.write(f"v {line[0]} {line[1]} {line[2]} {line[3]} {line[4]} {line[5]}\n")




import numpy as np


def update_obj_vertices(original_obj_path, new_vertices, output_obj_path):
    """
    Updates vertex positions in an OBJ file and saves it as a new OBJ file.

    Args:
    original_obj_path (str): Path to the original OBJ file.
    new_vertices (numpy.ndarray): Array of new vertex positions, shape (n, 3).
    output_obj_path (str): Path to save the updated OBJ file.
    """
    # Read the original OBJ file
    with open(original_obj_path, 'r') as file:
        lines = file.readlines()

    # Open the output file for writing
    with open(output_obj_path, 'w') as file:
        vertex_idx = 0
        for line in lines:
            if line.startswith('v '):
                # Replace the vertex position with the new one
                new_vertex = new_vertices[vertex_idx]
                file.write(f"v {new_vertex[0]} {new_vertex[1]} {new_vertex[2]}\n")
                vertex_idx += 1
            else:
                # Write the other lines unchanged
                file.write(line)



def grid_to_trimesh_faces(num_rows, num_cols):
    """
    Generates the triangle face indices required to convert a quad mesh into a trimesh based on the resolution of the quad mesh (number of rows and columns).
    Vertex indices are assumed to be arranged from right to left per row, and from top to bottom incrementally.

    Parameters:
      num_rows: int, the number of rows in the quad mesh
      num_cols: int, the number of columns in the quad mesh

    Returns:
      faces: numpy array, shape ((num_rows-1)*(num_cols-1)*2, 3), where each row represents the three vertex indices of a triangle.
    """
    faces = []
    for r in range(num_rows - 1):
        for c in range(num_cols - 1):
            # Calculate the indices of the four vertices of the current quad:
            v_tr = r * num_cols + c  # Top row, right vertex
            v_tl = r * num_cols + (c + 1)  # Top row, left vertex
            v_br = (r + 1) * num_cols + c  # Bottom row, right vertex
            v_bl = (r + 1) * num_cols + (c + 1)  # Bottom row, left vertex

            # Split the quad into two triangles
            faces.append([v_tr, v_br, v_bl])  # First triangle
            faces.append([v_tr, v_bl, v_tl])  # Second triangle
    return np.array(faces)


def loadJson(file_name):
    print(f"Loading {file_name}...", end="")
    with open(file_name) as json_file:
        dictionary = json.load(json_file)
        for key in ["camera_position", "camera_forward", "camera_up",  "optical_center", "focal_length", "image_size", "lower_left_corner", "upper_right_corner"]:
            if key in dictionary:
                dictionary[key] = np.array(dictionary[key])
    print("Done.")
    return dictionary


def visualize_acc(acc, channel_idx=0, v_min_max=None):
    if isinstance(acc, torch.Tensor):
        acc = acc.cpu().detach().numpy()
    acc = np.clip(acc[0, channel_idx], np.percentile(acc, 5), np.percentile(acc, 95))
    if v_min_max == None:
        vmin = np.min(acc)
        vmax = np.max(acc)
    else:
        vmin, vmax = v_min_max
    plt.imshow(acc.transpose(), vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.show()
    return vmin, vmax



if __name__ == '__main__':
    # Example usage:
    # new_vertices is an array of new vertex positions, shape (n, 3)
    new_vertices = np.random.rand(1024, 3)  # Example vertices
    new_vertices = torch.from_numpy(new_vertices).float()
    update_obj_vertices('/Users/ruochen/Documents/liris_code/Physics-guided_SfT/data/R1/0_R1_quad.obj', new_vertices, 'updated_mesh.obj')
