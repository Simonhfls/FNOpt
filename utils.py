import os

import torch
from torch.autograd import Function
from torch.nn.functional import normalize
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# "pseudo function" that doesn't affect the outputs but only scales the gradients

eps = 1e-12#1e-6#

class ScaleGrads(Function):
	@staticmethod
	def forward(input, scale):
		return input
	
	@staticmethod
	def setup_context(ctx, inputs, output):
		input, scale = inputs
		ctx.save_for_backward(scale)
	
	@staticmethod
	def backward(ctx, grad_output):
		scale = ctx.saved_tensors[0]
		return scale*grad_output, None # no gradients for gradient scaling

scale_grads = ScaleGrads.apply

# "pseudo function" that doesn't affect the outputs but only normalizes the gradients
class NormalizeGrads(Function):
	@staticmethod
	def forward(input):
		return input
	
	@staticmethod
	def setup_context(ctx, inputs, output):
		pass
		
	@staticmethod
	def backward(ctx, grad_output):
		#print(f"{grad_output.shape}: {grad_output.dtype}")
		# achtung, normalization kann sehr hohe / niedrige werte zurückgeben, wenn fast alles = 0 ist => min / max clampen
		return normalize(grad_output,dim=[i+1 for i in range(len(grad_output.shape)-1)],eps=1e-40).clamp(min=-10,max=10)
		
		#std = torch.mean(grad_output**2,dim=[i+1 for i in range(len(grad_output.shape)-1)]).detach().clamp_min(eps).reshape(*([grad_output.shape[0]]+[1 for i in range(len(grad_output.shape)-1)])) # bringt nichts (sollte das selbe wie normalize tun)
		#return grad_output/std

normalize_grads = NormalizeGrads.apply

# "pseudo function" that doesn't affect the outputs but only normalizes the gradients
class NormalizeGradsScale(Function):
	@staticmethod
	def forward(ctx, input, scale):
		ctx.scale = scale
		return input
	
	@staticmethod
	def backward(ctx, grad_output):
		# achtung, normalization kann sehr hohe / niedrige werte zurückgeben, wenn fast alles = 0 ist => min / max clampen
		return ctx.scale*normalize(grad_output,dim=[i+1 for i in range(len(grad_output.shape)-1)]).clamp(min=-10,max=10), None
		
		#std = torch.mean(grad_output**2,dim=[i+1 for i in range(len(grad_output.shape)-1)]).detach().clamp_min(eps).reshape(*([grad_output.shape[0]]+[1 for i in range(len(grad_output.shape)-1)])) # bringt nichts (sollte das selbe wie normalize tun)
		#return grad_output/std

normalize_grads_scale = NormalizeGradsScale.apply


def log_range_params(range_params,default_param=1):# useful to sample parameters from "exponential distribution"
	range_params = default_param if range_params is None else range_params
	range_params = [range_params,range_params] if type(range_params) is not list else range_params
	range_params = np.log(range_params)
	return range_params[0],range_params[1]-range_params[0]

def range_params(r_params,default_param=1):# useful to sample parameters from "exponential distribution"
	r_params = default_param if r_params is None else r_params
	r_params = [r_params,r_params] if type(r_params) is not list else r_params
	return r_params[0],r_params[1]-r_params[0]

def has_nan(x):
	if type(x) is not torch.Tensor:
		return False
	return torch.any(x.isnan())

def has_inf(x):
	if type(x) is not torch.Tensor:
		return False
	return torch.any(x.isinf())

def value_range(x):
	if type(x) is not torch.Tensor:
		return None
	return [torch.min(x).detach().cpu().numpy(), torch.max(x).detach().cpu().numpy()]



####################OTHERS
def grid_to_triangular_mesh(grid):
	"""
    Transform a quad grid into a triangular mesh.

    Parameters:
    - grid (torch.Tensor): Input tensor of shape (bs, H, W, 3), representing 3D vertex coordinates.

    Returns:
    - verts (torch.Tensor): Tensor of shape (bs, V, 3), where V = H * W.
    - faces (torch.Tensor): Tensor of shape (F, 3), where F = (H - 1) * (W - 1) * 2.
      The faces are the same for each mesh in the batch.
    """
	bs, H, W, C = grid.shape
	assert C == 3, "Input grid must have 3 channels (x, y, z)"

	# Reshape grid into vertices for each mesh in the batch
	verts = grid.reshape(bs, H * W, 3)  # (bs, H*W, 3)

	# Generate face indices for a single mesh of size (H, W)
	# Create a grid of indices for the top-left corner of each quad
	i = torch.arange(H - 1)
	j = torch.arange(W - 1)
	ii, jj = torch.meshgrid(i, j, indexing="ij")  # shape: (H-1, W-1)

	# Compute indices for the four corners of each quad
	v0 = ii * W + jj  # top-left corner
	v1 = v0 + 1  # top-right corner
	v2 = v0 + W  # bottom-left corner
	v3 = v2 + 1  # bottom-right corner

	# Form two triangles for each quad:
	# First triangle: (v0, v2, v1)
	tri1 = torch.stack([v0, v2, v1], dim=-1).reshape(-1, 3)
	# Second triangle: (v1, v2, v3)
	tri2 = torch.stack([v1, v2, v3], dim=-1).reshape(-1, 3)

	# Concatenate both triangle sets to form the full face tensor
	faces = torch.cat([tri1, tri2], dim=0)  # (F, 3)

	return verts, faces


import numpy as np

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

def generate_ffmpeg_cmd(
    render_dir,
    output_file,
    framerate,
    n_frames,
	output_dir=None,
    input_pattern="*.png",
    codec="libx264",
    pixel_format="yuv420p"
):
    """
    Generate an ffmpeg command to convert image sequence to video.

    Args:
        render_dir (str): Directory containing input images.
        output_file (str): Output video file name (relative or absolute path).
        framerate (int): Video frame rate.
        n_frames (int): Number of frames to include in the video.
        input_pattern (str, optional): Glob pattern for input images. Defaults to '*.png'.
        codec (str, optional): Codec for video encoding. Defaults to 'libx264'.
        pixel_format (str, optional): Pixel format. Defaults to 'yuv420p'.

    Returns:
        list: The ffmpeg command as a list of arguments.
    """
    if output_dir is None:
        output_dir = render_dir
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output files without asking
        "-framerate", str(framerate),
        "-pattern_type", "glob",
        "-i", os.path.join(render_dir, input_pattern),
        "-frames:v", str(n_frames),
        "-c:v", codec,
        "-pix_fmt", pixel_format,
        os.path.join(output_dir, output_file)
    ]
    return cmd

import re

def get_unique_filename(base_name, output_dir, ext=".mp4"):
    """
    Returns a unique filename by appending _001, _002, ... if needed.
    It detects and strips existing _### suffix before adding a new one.
    """
    # Split name and extension
    base, extension = os.path.splitext(base_name)
    # Remove any existing _NNN suffix
    base = re.sub(r'_\d{3}$', '', base)

    counter = 1
    candidate = f"{base}{extension}"
    while os.path.exists(os.path.join(output_dir, candidate)):
        candidate = f"{base}_{counter:03d}{extension}"
        counter += 1
    return candidate


def get_face_areas_batch(vertices, faces):
    '''
    Computes the area of each face in a batch of meshes
    vertices: (batch_size, num_vertices, 3)
    faces: (num_faces, 3)

    returns: (batch_size, num_faces)
    '''
    v1 = vertices[:, faces[:, 0], :]  # (batch_size, num_faces, 3)
    v2 = vertices[:, faces[:, 1], :]  # (batch_size, num_faces, 3)
    v3 = vertices[:, faces[:, 2], :]  # (batch_size, num_faces, 3)
    edge1 = v2 - v1
    edge2 = v3 - v1
    cross_product = torch.cross(edge1, edge2, dim=2)  # (batch_size, num_faces, 3)
    area = torch.norm(cross_product, dim=2) / 2  # (batch_size, num_faces)
    return area

if __name__=='__main__':
	# test grid_to_triangular_mesh
	batch_size = 2
	height = 4
	width = 4
	grid = torch.randn(batch_size, height, width, 3)
	verts, faces1 = grid_to_triangular_mesh(grid)
	print("verts:", verts.shape)
	print("faces:", faces1.shape)
	# test grid_to_trimesh_faces
	num_rows = 4
	num_cols = 4
	faces2 = grid_to_trimesh_faces(num_rows, num_cols)
	print("faces:", faces2.shape)
