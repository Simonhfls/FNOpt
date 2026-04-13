import math
import os
import sys

from dataset_utils import rotation_matrix
from losses.repulsive_energy import RepulsiveEnergy

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
from torch import vmap
import torch.nn.functional as F
from get_param import params,toCuda, device
from utils import log_range_params, range_params, grid_to_trimesh_faces, get_f_connectivity_edges
import wandb

eps = 1e-7
step = 0
resolution_tpe = params.cloth.repulsive.resolution

f = grid_to_trimesh_faces(num_rows=resolution_tpe, num_cols=resolution_tpe)
f_connectivity_edges = get_f_connectivity_edges(torch.from_numpy(f).to(device))
repulsive_loss = RepulsiveEnergy(threshold=params.cloth.repulsive.thres)

"""
ask-tell interface:
ask(): ask for batch of gradients of velocities wrt certain loss function
tell(): tell update step for velocities (positions are updated internally) => return loss to update NN parameters
"""
#Attention: x/y are swapped (x-dimension=1; y-dimension=0)
# n_vertices = params.data.height*params.data.width
L_0 = params.cloth.L_0
dt = params.cloth.dt
k_repulsive = params.cloth.repulsive.k

def loss(x_old, v_old, acc, force, bc_masks, bc_positions, M, stiffnesses, shearings, bendings, k_repulsive=0, f_connectivity_edges=None, repulsive_loss=None):
	"""
	:return:
		:loss: loss values for samples in batch (shape: batch_size)
		:E_int: internal energies for samples in batch (shape: batch_size)
	"""
	n_vertices = x_old.shape[-1] * x_old.shape[-2]

	global step
	# integrate velocity and positions
	v_new = v_old + dt * acc
	x_new = x_old + dt * v_new

	# apply boundary conditions
	x_new = bc_masks * bc_positions + (1 - bc_masks) * x_new
	# v_new = (1 - bc_masks) * v_new

	# compute energy terms
	dx_i = x_new[:, :, 1:] - x_new[:, :, :-1]
	dx_j = x_new[:, :, :, 1:] - x_new[:, :, :, :-1]

	dx_n_i = torch.nn.functional.normalize(dx_i, dim=1)
	dx_n_j = torch.nn.functional.normalize(dx_j, dim=1)

	# # stiffness energy
	stiffness_i = torch.mean((torch.sqrt(torch.sum(dx_i[:, :3] ** 2, 1)) - L_0) ** 2, [1, 2])
	stiffness_j = torch.mean((torch.sqrt(torch.sum(dx_j[:, :3] ** 2, 1)) - L_0) ** 2, [1, 2])
	M_Stiff = (stiffness_i + stiffness_j)
	E_stiff = stiffnesses * M_Stiff

	# # Davids version of shearing energy
	angle_1 = torch.arccos(
		torch.einsum('abcd,abcd->acd', dx_n_i[:, :, :, :-1], dx_n_j[:, :, :-1]).clamp(eps - 1, 1 - eps))
	angle_2 = torch.arccos(
		torch.einsum('abcd,abcd->acd', dx_n_i[:, :, :, :-1], dx_n_j[:, :, 1:]).clamp(eps - 1, 1 - eps))
	angle_3 = torch.arccos(
		torch.einsum('abcd,abcd->acd', dx_n_i[:, :, :, 1:], dx_n_j[:, :, :-1]).clamp(eps - 1, 1 - eps))
	angle_4 = torch.arccos(
		torch.einsum('abcd,abcd->acd', dx_n_i[:, :, :, 1:], dx_n_j[:, :, 1:]).clamp(eps - 1, 1 - eps))
	M_shear = (torch.sum((angle_1 - torch.pi / 2) ** 2, [1, 2])
			   + torch.sum((angle_2 - torch.pi / 2) ** 2, [1, 2])
			   + torch.sum((angle_3 - torch.pi / 2) ** 2, [1, 2])
			   + torch.sum((angle_4 - torch.pi / 2) ** 2, [1, 2]))
	E_shear = shearings * M_shear / n_vertices


	bend_1 = torch.arccos(torch.einsum('abcd,abcd->acd', dx_n_i[:, :, 1:], dx_n_i[:, :, :-1]).clamp(eps - 1, 1 - eps))
	bend_2 = torch.arccos(
		torch.einsum('abcd,abcd->acd', dx_n_j[:, :, :, 1:], dx_n_j[:, :, :, :-1]).clamp(eps - 1, 1 - eps))
	M_bend = torch.sum((bend_1 - 0) ** 2, [1, 2]) + torch.sum((bend_2 - 0) ** 2, [1, 2])
	E_bend = bendings * M_bend / n_vertices

	# inertia term
	L_inert = 0.5 * torch.mean(torch.sum(M * acc ** 2, dim=1), [1, 2]) * dt ** 2

	# external forces term
	L_ext = -torch.mean(torch.einsum('abcd,abcd->acd', acc, force * M), [1, 2]) * dt ** 2
	bs = x_new.shape[0]
	L_repulsive = toCuda(torch.zeros(x_new.shape[0]))

	loss_dict = {}

	if k_repulsive > 0:
		# subsample vertices
		x_subsampled = F.interpolate(x_new, size=(resolution_tpe, resolution_tpe), mode='bilinear', align_corners=True)
		x_subsampled = x_subsampled.permute(0, 2, 3, 1).reshape(1, -1, 3)

		M_repulsive = repulsive_loss(x_subsampled, f_connectivity_edges)

		L_repulsive = k_repulsive * M_repulsive
		loss_dict["L_repulsive"] = L_repulsive
		loss_dict["M_repulsive"] = M_repulsive

	E_int = E_stiff + E_shear + E_bend
	L = E_int + L_ext + L_inert + L_repulsive
	loss_dict.update({
		"L": L, "L_stiff": E_stiff,
		"L_shear": E_shear, "L_bend": E_bend,
		"L_ext": L_ext, "L_inert": L_inert,
		"E_int": E_int,
		"M_stiff": M_Stiff, "M_shear": M_shear, "M_bend": M_bend
	})

	step += 1
	return loss_dict

@torch.jit.script
def _make_kernels(C: int, device: torch.device):
	kv = torch.tensor([[-1., 1.]], device=device)
	kv = kv.view(1,1,2,1).repeat(C,1,1,1)
	kh = torch.tensor([[-1.],[1.]], device=device)
	kh = kh.view(1,1,1,2).repeat(C,1,1,1)
	return kv, kh


# @torch.jit.script
def loss_efficient(x_old, v_old, acc, force, bc_masks, bc_positions, M, stiffnesses, shearings, bendings,
				   k_repulsive,
				   f_connectivity_edges):
	"""
	A more efficient implementation of loss functions.
	:return:
		:loss: loss values for samples in batch (shape: batch_size)
		:E_int: internal energies for samples in batch (shape: batch_size)
	"""

	# global step
	# integrate velocity and positions
	dt = 1
	L_0 = 1
	eps = 1e-7
	v_new = v_old + dt * acc
	x_new = x_old + dt * v_new
	n_vertices = x_old.shape[-1] * x_old.shape[-2]
	B, C, H, W = x_new.shape

	# apply boundary conditions
	x_new = bc_masks * bc_positions + (1 - bc_masks) * x_new

	kernel_v, kernel_h = _make_kernels(C, x_new.device)
	dx_i = F.conv2d(x_new, weight=kernel_v, groups=C)
	dx_j = F.conv2d(x_new, weight=kernel_h, groups=C)

	dx_n_i = torch.nn.functional.normalize(dx_i, dim=1)
	dx_n_j = torch.nn.functional.normalize(dx_j, dim=1)

	# stiffness energy
	norm_i = torch.linalg.norm(dx_i[:, :3], dim=1)
	norm_j = torch.linalg.norm(dx_j[:, :3], dim=1)

	dev_i = (norm_i - L_0).pow(2)  # (B, H', W')
	dev_j = (norm_j - L_0).pow(2)  # (B, H, W')
	M_Stiff = dev_i.mean(dim=[1, 2]) + dev_j.mean(dim=[1, 2])
	E_stiff = stiffnesses * M_Stiff

	# Davids version of shearing energy (More efficient version)
	a1 = dx_n_i[..., :-1]  # drop last col → (B,3,31,31)
	b1 = dx_n_j[..., :-1, :]  # drop last row → (B,3,31,31)

	a2 = dx_n_i[..., :-1]  # same
	b2 = dx_n_j[..., 1:, :]  # drop first row → (B,3,31,31)

	a3 = dx_n_i[..., 1:]  # drop first col → (B,3,31,31)
	b3 = dx_n_j[..., :-1, :]  # drop last row → (B,3,31,31)

	a4 = dx_n_i[..., 1:]  # drop first col → (B,3,31,31)
	b4 = dx_n_j[..., 1:, :]  # drop first row → (B,3,31,31)

	A = torch.stack([a1, a2, a3, a4], dim=0)  # (4, B, C, H', W')
	B = torch.stack([b1, b2, b3, b4], dim=0)  # (4, B, C, H', W')
	dot = (A * B).sum(dim=2)

	ang = torch.acos(dot.clamp(eps - 1, 1 - eps))
	dev = (ang - (math.pi / 2)).pow(2)
	M_shear = dev.sum(dim=0).sum(dim=[1, 2])
	E_shear = shearings * M_shear / n_vertices

	# Davids version of bending energy (Efficient version)
	bi1 = dx_n_i[:, :, 1:, :]  # (B,C,H-1,W)
	bj1 = dx_n_i[:, :, :-1, :]  # (B,C,H-1,W)
	dot1 = (bi1 * bj1).sum(dim=1)  # (B, H-1, W)
	ang1 = torch.acos(dot1.clamp(eps - 1, 1 - eps))  # (B, H-1, W)
	m1 = ang1.pow(2).sum(dim=[1, 2])  # (B,)

	bi2 = dx_n_j[:, :, :, 1:]  # (B,C,H,W-1)
	bj2 = dx_n_j[:, :, :, :-1]  # (B,C,H,W-1)
	dot2 = (bi2 * bj2).sum(dim=1)  # (B, H, W-1)
	ang2 = torch.acos(dot2.clamp(eps - 1, 1 - eps))  # (B, H, W-1)
	m2 = ang2.pow(2).sum(dim=[1, 2])  # (B,)

	M_bend = m1 + m2  # (B,)
	E_bend = bendings * M_bend / n_vertices

	# compute inertia term
	L_inert = 0.5 * torch.mean(torch.sum(M * acc ** 2, dim=1), [1, 2]) * dt ** 2

	# compute external forces term
	L_ext = -torch.mean(torch.einsum('abcd,abcd->acd', acc, force * M), [1, 2]) * dt ** 2

	loss_dict = {}
	L = torch.zeros(x_new.shape[0], device=x_new.device)
	if k_repulsive > 0:
		# subsample vertices
		x_subsampled = F.interpolate(x_new, size=(32, 32), mode='bilinear', align_corners=True)
		x_subsampled = x_subsampled.permute(0, 2, 3, 1).reshape(1, -1, 3)
		M_repulsive = repulsive_loss(x_subsampled, f_connectivity_edges)
		L_repulsive = k_repulsive * M_repulsive
		loss_dict["L_repulsive"] = L_repulsive
		L = L + L_repulsive

	E_int = E_stiff + E_shear + E_bend
	L = L + E_int + L_ext + L_inert
	# if k_repulsive > 0:
	# 	L = L + L_repulsive
	loss_dict['L'] = L

	# step += 1
	return loss_dict



@torch.jit.script
def loss_efficient_jit(x_old, v_old, acc, force, bc_masks,
					   bc_positions, M, stiffnesses, shearings, bendings,
					   k_repulsive: float=0,
					   f_connectivity_edges: torch.Tensor=None,
					   resolution_tpe: int=32
					   ):
	"""
	A more efficient implementation of loss functions using jit.
	:return:
		:loss: loss values for samples in batch (shape: batch_size)
		:E_int: internal energies for samples in batch (shape: batch_size)
	"""

	# global step
	# integrate velocity and positions
	dt = 1
	L_0 = 1
	eps = 1e-7
	v_new = v_old + dt * acc
	x_new = x_old + dt * v_new
	n_vertices = x_old.shape[-1] * x_old.shape[-2]
	B, C, H, W = x_new.shape

	# apply boundary conditions
	x_new = bc_masks * bc_positions + (1 - bc_masks) * x_new

	kernel_v, kernel_h = _make_kernels(C, x_new.device)
	dx_i = F.conv2d(x_new, weight=kernel_v, groups=C)
	dx_j = F.conv2d(x_new, weight=kernel_h, groups=C)

	dx_n_i = torch.nn.functional.normalize(dx_i, dim=1)
	dx_n_j = torch.nn.functional.normalize(dx_j, dim=1)

	# stiffness energy
	norm_i = torch.linalg.norm(dx_i[:, :3], dim=1)
	norm_j = torch.linalg.norm(dx_j[:, :3], dim=1)

	dev_i = (norm_i - L_0).pow(2)  # (B, H', W')
	dev_j = (norm_j - L_0).pow(2)  # (B, H, W')
	M_Stiff = dev_i.mean(dim=[1, 2]) + dev_j.mean(dim=[1, 2])
	E_stiff = stiffnesses * M_Stiff

	# Davids version of shearing energy (More efficient version)
	a1 = dx_n_i[..., :-1]  # drop last col → (B,3,31,31)
	b1 = dx_n_j[..., :-1, :]  # drop last row → (B,3,31,31)

	a2 = dx_n_i[..., :-1]  # same
	b2 = dx_n_j[..., 1:, :]  # drop first row → (B,3,31,31)

	a3 = dx_n_i[..., 1:]  # drop first col → (B,3,31,31)
	b3 = dx_n_j[..., :-1, :]  # drop last row → (B,3,31,31)

	a4 = dx_n_i[..., 1:]  # drop first col → (B,3,31,31)
	b4 = dx_n_j[..., 1:, :]  # drop first row → (B,3,31,31)

	A = torch.stack([a1, a2, a3, a4], dim=0)  # (4, B, C, H', W')
	B = torch.stack([b1, b2, b3, b4], dim=0)  # (4, B, C, H', W')
	dot = (A * B).sum(dim=2)

	ang = torch.acos(dot.clamp(eps - 1, 1 - eps))
	dev = (ang - (math.pi / 2)).pow(2)
	M_shear = dev.sum(dim=0).sum(dim=[1, 2])
	E_shear = shearings * M_shear / n_vertices

	# Davids version of bending energy (Efficient version)
	bi1 = dx_n_i[:, :, 1:, :]  # (B,C,H-1,W)
	bj1 = dx_n_i[:, :, :-1, :]  # (B,C,H-1,W)
	dot1 = (bi1 * bj1).sum(dim=1)  # (B, H-1, W)
	ang1 = torch.acos(dot1.clamp(eps - 1, 1 - eps))  # (B, H-1, W)
	m1 = ang1.pow(2).sum(dim=[1, 2])  # (B,)

	bi2 = dx_n_j[:, :, :, 1:]  # (B,C,H,W-1)
	bj2 = dx_n_j[:, :, :, :-1]  # (B,C,H,W-1)
	dot2 = (bi2 * bj2).sum(dim=1)  # (B, H, W-1)
	ang2 = torch.acos(dot2.clamp(eps - 1, 1 - eps))  # (B, H, W-1)
	m2 = ang2.pow(2).sum(dim=[1, 2])  # (B,)

	M_bend = m1 + m2  # (B,)
	E_bend = bendings * M_bend / n_vertices

	# compute inertia term
	L_inert = 0.5 * torch.mean(torch.sum(M * acc ** 2, dim=1), [1, 2]) * dt ** 2

	# compute external forces term
	L_ext = -torch.mean(torch.einsum('abcd,abcd->acd', acc, force * M), [1, 2]) * dt ** 2

	loss_dict = {}
	L = torch.zeros(x_new.shape[0], device=x_new.device)
	if k_repulsive > 0:
		# subsample vertices
		x_subsampled = F.interpolate(x_new, size=(resolution_tpe, resolution_tpe), mode='bilinear', align_corners=True)
		x_subsampled = x_subsampled.permute(0, 2, 3, 1).reshape(1, -1, 3)
		M_repulsive = repulsive_loss(x_subsampled, f_connectivity_edges)
		L_repulsive = k_repulsive * M_repulsive
		loss_dict["L_repulsive"] = L_repulsive
		L = L + L_repulsive

	E_int = E_stiff + E_shear + E_bend
	L = L + E_int + L_ext + L_inert

	loss_dict['L'] = L

	# step += 1
	return loss_dict



class DatasetCloth:
	def __init__(self, h, w, batch_size=100, dataset_size=1000, average_sequence_length=5000, stiffness_range=None,
				 shearing_range=None, bending_range=None, a_ext_range=None, a_ext_noise_range=0,
				 iterations_per_timestep=5):

		# dataset parameters
		self.h, self.w = h, w
		self.batch_size = batch_size
		self.dataset_size = dataset_size
		self.average_sequence_length = average_sequence_length

		# grid utility
		x_space = torch.linspace(0, L_0 * (w - 1), w)
		y_space = torch.linspace(-L_0 * (h - 1) / 2, L_0 * (h - 1) / 2, h)
		y_grid, x_grid = torch.meshgrid(y_space, x_space, indexing="ij")
		self.y_mesh, self.x_mesh = toCuda(torch.meshgrid([torch.arange(0, self.h), torch.arange(0, self.w)]))

		# cloth state values
		self.x_0 = toCuda(torch.cat([x_grid.unsqueeze(0), y_grid.unsqueeze(0), torch.zeros(1, h, w)], dim=0))
		self.v_0 = toCuda(torch.zeros(3, h, w))
		self.x = toCuda(torch.zeros(self.dataset_size, 3, self.h, self.w))  # positions
		self.v = toCuda(torch.zeros(self.dataset_size, 3, self.h, self.w))  # velocities
		self.a = toCuda(torch.zeros(dataset_size, 3, h, w))  # accelerations (start at zero and get updated for every iteration until next timestep)
		self.T = toCuda(torch.zeros(self.dataset_size, 1))  # timestep
		self.iterations = toCuda(torch.zeros(dataset_size))  # iterations for the individual training pool samples
		self.iterations_per_timestep = iterations_per_timestep  # number of iterations per timestep

		self.hidden_states = [None for _ in range(dataset_size)]

		# simulation / cloth parameters
		self.M = torch.ones(1, 1, h, w, device=device)  # Mass matrix TODO change according to cloth resolution
		self.M[:, :, 0] = self.M[:, :, -1] = self.M[:, :, :, 0] = self.M[:, :, :, -1] = 0.5
		self.M[:, :, 0, 0] = self.M[:, :, 0, -1] = self.M[:, :, -1, 0] = self.M[:, :, -1, -1] = 0.25

		self.stiffness_range = log_range_params(stiffness_range)
		self.shearing_range = log_range_params(shearing_range)
		self.bending_range = log_range_params(bending_range)

		self.g_vect = toCuda(torch.tensor([0, 0, -1.])).unsqueeze(0).repeat(self.dataset_size, 1).unsqueeze(
			2).unsqueeze(3)  # gravity vector. CODO: radnom directions / strengths of gravity
		self.a_ext_range = range_params(a_ext_range)
		self.a_exts = toCuda(torch.ones(self.dataset_size, 3, self.h, self.w)) * self.g_vect  # external forces
		self.a_exts_damping = 0.999
		self.da_exts_dt = toCuda(torch.zeros(self.dataset_size, 3, self.h, self.w))  # derivatives of external forces
		self.da_exts_dt_damping = 0.95
		self.a_ext_noise_range = a_ext_noise_range

		self.rot_speed = toCuda(
			torch.zeros(self.dataset_size, 3, 3))  # delta rotation matrix that is recurrently multiplied onto rotations
		self.translation_freq = toCuda(torch.zeros(self.dataset_size, 3, 1,
												   1))  # delta rotation matrix that is recurrently multiplied onto rotations
		self.pinch_freq = toCuda(torch.zeros(self.dataset_size, 1, 1,
											 1))  # delta rotation matrix that is recurrently multiplied onto rotations
		self.translation_amp = toCuda(torch.zeros(self.dataset_size, 3, 1,
												  1))  # delta rotation matrix that is recurrently multiplied onto rotations
		self.rotations = toCuda(torch.zeros(self.dataset_size, 3, 3))
		self.bc_positions = toCuda(
			torch.zeros(self.dataset_size, 3, self.h, self.w))  # positions for boundary conditions

		# positions for boundary conditions without scaling (pinching) and translations
		self.bc_positions_orig = toCuda(torch.zeros(self.dataset_size, 3, self.h, self.w))
		self.bc_masks = toCuda(
			torch.zeros(self.dataset_size, 1, self.h, self.w))  # binary mask, where to apply bc_positions
		# self.E_repulsive = toCuda(torch.zeros(self.dataset_size))  # repulsive energy
		# compute initial repulsive energy
		x_subsampled = F.interpolate(self.x_0.unsqueeze(0), size=(resolution_tpe, resolution_tpe), mode='bilinear', align_corners=True)
		x_subsampled = x_subsampled.permute(0,2,3,1).reshape(-1, 3)

		"""
		self.stiffnesses = toCuda(torch.zeros(self.dataset_size))
		self.shearings = toCuda(torch.zeros(self.dataset_size))
		self.bendings = toCuda(torch.zeros(self.dataset_size))
		"""
		# set material parameters once and don't change during reset to avoid bias towards simpler parameters
		self.stiffnesses = toCuda(
			torch.exp(self.stiffness_range[0] + torch.rand(self.dataset_size) * self.stiffness_range[1]))  # 1000#
		self.shearings = toCuda(
			torch.exp(self.shearing_range[0] + torch.rand(self.dataset_size) * self.shearing_range[1]))  # 10#0#
		self.bendings = toCuda(
			torch.exp(self.bending_range[0] + torch.rand(self.dataset_size) * self.bending_range[1]))  # 0#10#

		# self.a_exts = torch.ones(self.dataset_size,3,self.h,self.w)*torch.tensor([0,0,-1]).unsqueeze(0).unsqueeze(2).unsqueeze(3) # init with gravity. CODO: radnom directions / strengths of gravity


		for i in range(self.dataset_size):
			self.reset_env(i)

		self.step = 0  # number of tell()-calls
		self.reset_i = 0

		# for efficient computing
		self._asked_grads = torch.zeros(
			(batch_size, 3, h, w),
			device=device,
			requires_grad=True
		)

	def reset_env(self, index):

		# print(f"reset {index}")

		# material parameters
		"""
		self.stiffnesses[index] = torch.exp(self.stiffness_range[0]+torch.rand(1)*self.stiffness_range[1])#1000#
		self.shearings[index] = torch.exp(self.shearing_range[0]+torch.rand(1)*self.shearing_range[1])#10#0#
		self.bendings[index] = torch.exp(self.bending_range[0]+torch.rand(1)*self.bending_range[1])#0#10#
		"""

		# initial rotation of cloth
		yaw = (torch.rand(1) - 0.5) * 2 * 2 * 3.14  # 0#
		pitch = (torch.rand(1) - 0.5) * 2 * 2 * 3.14  # 0#
		roll = (torch.rand(1) - 0.5) * 2 * 2 * 3.14  # 0#
		self.rotations[index] = rotation_matrix(yaw, pitch, roll, device=device)

		# rotation speed for boundary conditions
		moving = 1 if torch.rand(1) < 0.8 else 0
		dyaw = moving * (torch.rand(1) - 0.5) * 2 * 2 * 3.14 * 0.01
		dpitch = moving * (torch.rand(1) - 0.5) * 2 * 2 * 3.14 * 0.01
		droll = moving * (torch.rand(1) - 0.5) * 2 * 2 * 3.14 * 0.01  # keep only roll for rotation
		self.rot_speed[index] = rotation_matrix(dyaw, dpitch, droll, device=device)

		# translation speeds + pinch movement for boundary conditions
		self.translation_freq[index] = moving * (torch.rand(3, 1, 1) - 0.5) * 2 * 0.2  # 0#
		self.translation_amp[index] = moving * (torch.rand(3, 1, 1) - 0.5) * 2 * 10  # 0#
		self.pinch_freq[index] = moving * (torch.rand(1, 1, 1) - 0.5) * 2 * 0.2  # 0#



		# reset state
		self.hidden_states[index] = None  # hidden state for (neural) optimizer
		self.T[index] = 0  # time of env
		self.x[index] = torch.einsum("ab,bcd->acd", self.rotations[index], self.x_0.clone())
		self.v[index] = self.v_0.clone()
		self.a[index] = 0
		self.iterations[index] = 0

		# boundary conditions
		self.bc_masks[index] = 0
		self.bc_masks[index, :, 0, 0] = 1
		self.bc_masks[index, :, -1, 0] = 1
		while torch.rand(1) < 0.5:  # add further random bc
			self.bc_masks[index, :, torch.randint(0, self.h, [1]), torch.randint(0, self.w, [1])] = 1
		self.bc_positions[index] = self.x[index].clone()
		self.bc_positions_orig[index] = self.x[index].clone()

		# external forces

		# self.a_exts[index] = torch.exp(self.a_ext_range[0]+torch.rand(1)*self.a_ext_range[1]) # TODO: init with gravity
		g_scale = self.a_ext_range[0] + torch.rand(1, device=device) * self.a_ext_range[1]
		# self.g_vect[index,:,0,0] = torch.tensor([0,0,-1.0],device=device)*g_scale#
		self.g_vect[index, :, 0, 0] = torch.einsum("ab,b->a", rotation_matrix((torch.rand(1) - 0.5) * 2 * 2 * 3.14,
																			  (torch.rand(1) - 0.5) * 2 * 2 * 3.14,
																			  (torch.rand(1) - 0.5) * 2 * 2 * 3.14,
																			  device=device),
												   torch.tensor([0, 0, -1.0], device=device) * g_scale)
		self.a_exts[index, :, :, :] = self.g_vect[index]
		# print(f"rot mat: {rotation_matrix((torch.rand(1)-0.5)*2*2*3.14,(torch.rand(1)-0.5)*2*2*3.14,(torch.rand(1)-0.5)*2*2*3.14)}")
		# print(f"g_vect: {self.g_vect[index,:,0,0]}")
		self.da_exts_dt[index, :, :, :] = 0
		# compute original repulsive energy
		# self.E_repulsive[index] = self.init_E_repulsive


	def reset0_inference_env(self, x_0=None, bc_indices=None):
		self.indices = torch.zeros(1, dtype=torch.long, device=self.x.device)
		x_0 = x_0 if x_0 is not None else self.x_0
		# material parameters
		index = 0
		# initial rotation of cloth
		self.rotations[index] = rotation_matrix(0, 0, 0, device=device)

		# rotation speed for boundary conditions
		self.rot_speed[index] = rotation_matrix(0, 0, 0, device=device)

		# translation speeds + pinch movement for boundary conditions
		self.translation_freq[index] = 0  # (torch.rand(3)-0.5)*2*0.2
		self.translation_amp[index] = 0  # (torch.rand(3)-0.5)*2*10
		self.pinch_freq[index] = 0  # (torch.rand(1)-0.5)*2*0.2

		# reset state
		self.hidden_states[index] = None  # hidden state for (neural) optimizer
		self.T[index] = 0  # time of env
		self.x[index] = x_0.clone()
		self.v[index] = self.v_0.clone()
		self.a[index] = 0

		# boundary conditions
		self.bc_masks[index] = 0
		if bc_indices is None:
			# hang points at the corners
			self.bc_masks[index, :, 0, 0] = 1
			self.bc_masks[index, :, -1, 0] = 1
		else:
			# bc_indices: [[a1, b1], [a2, b2], ...]
			for a, b in bc_indices:
				self.bc_masks[index, :, a, b] = 1
		# self.bc_masks[index,:,0,-1] = 1
		# self.bc_masks[index,:,-1,-1] = 1
		# self.bc_masks[index, :, self.h // 2, self.w // 2] = 1		# hang points at the center
		# self.bc_masks[index,:,self.h//5,self.w//5] = 1
		self.bc_positions[index] = self.x[index].clone()		# (1, 3, h, w)
		self.bc_positions_orig[index] = self.x[index].clone()
		self.g_vect[index, :, 0, 0] = torch.tensor([0, 0, -1], device=device)



	def set_position(self, rest_position):
		# self.x[index] = rest_position.clone()
		self.x = rest_position.clone().unsqueeze(0)

	def set_bc_positions(self, bc_positions):
		# set boundary conditions
		self.bc_positions = bc_positions.unsqueeze(0)

	def set_materials(self, stretch, shear, bend):
		self.stiffnesses = stretch.unsqueeze(0)		# (1,)
		self.shearings = shear.unsqueeze(0)
		self.bendings = bend.unsqueeze(0)

	def set_external_forces(self, a_exts):
		self.a_exts = a_exts								# (1, 3, h, w)

	def set_acc(self, a):
		self.a = a								# (n_frames, 3, h, w)


	def set_mass(self, m):
		self.M = m * torch.ones(1, 1, self.h, self.w, device=device)  # Mass matrix TODO change according to cloth resolution
		self.M[:, :, 0] = self.M[:, :, -1] = self.M[:, :, :, 0] = self.M[:, :, :, -1] = 0.5 * m
		self.M[:, :, 0, 0] = self.M[:, :, 0, -1] = self.M[:, :, -1, 0] = self.M[:, :, -1, -1] = 0.25 * m


	def set_optimizable(self, a_ext, stretch, shear, bend):
			# set external forces
			self.set_external_forces(a_ext)						# (1, 3, h, w)
			self.set_materials(stretch, shear, bend)

	def update_env(self, index, bc_velocity=None, frame_counter=None):
		a_index = index if frame_counter is None else frame_counter
		# update state
		self.v[index] += self.a[a_index] * dt
		self.x[index] += self.v[index] * dt
		if frame_counter is None:
			self.a[a_index] = 0

		if bc_velocity is None:
			# update boundary conditions
			self.bc_positions_orig[index] = torch.einsum("ab,bcd->acd", self.rot_speed[index],
														 self.bc_positions_orig[index])
			self.bc_positions[index] = self.bc_positions_orig[index] * (
						torch.cos(self.T[index] * self.pinch_freq[index]) * 0.4 + 0.6) + torch.sin(
				self.T[index] * self.translation_freq[index]) * self.translation_amp[index]

			# apply boundary conditions
			self.x[index] = self.bc_masks[index] * self.bc_positions[index] + (1 - self.bc_masks[index]) * self.x[index]
			self.v[index] = (1 - self.bc_masks[index]) * self.v[index]
		else:
			# update boundary conditions
			# self.bc_positions[index] = bc_positions
			self.x[index] = self.bc_masks[index] * self.bc_positions[index] + (1 - self.bc_masks[index]) * self.x[index]
			self.v[index] = torch.where(self.bc_masks[index].bool(), bc_velocity, self.v[index])		# set bc velocities

		# TODO update external forces
		if params.training:
			if params.cloth.use_f_ext:
				# update external forces (CODO: clip min/max forces) ...not very efficient (slows down test_cv2_interactive by approx 10%)
				self.a_exts[self.indices,:,:,:] = self.a_exts_damping*self.a_exts[self.indices,:,:,:]+(1-self.a_exts_damping)*self.g_vect[self.indices]+0.01*self.da_exts_dt[self.indices,:,:,:]
				if torch.rand(1)<0.3:
					gaussian_w = toCuda((torch.rand(1)*30)**2)
					gaussian = torch.exp(-((self.x_mesh-toCuda(torch.rand(1,1))*self.w)**2+(self.y_mesh-toCuda(torch.rand(1,1))*self.h)**2)/gaussian_w).unsqueeze(0).unsqueeze(1)
					gaussian = gaussian*toCuda(torch.randn(1,3,1,1))
				else:
					gaussian = 0
				self.da_exts_dt[self.indices,:,:,:] = self.da_exts_dt_damping*self.da_exts_dt[self.indices,:,:,:]+0.1*toCuda(torch.randn(1,3,1,1))+gaussian

				# add random noise to a_exts
				# a_ext_noise = self.a_ext_noise_range*torch.rand(self.batch_size).unsqueeze(1).unsqueeze(2).unsqueeze(3)*torch.randn(self.batch_size,3,self.h,self.w)

			else:
				self.a_exts[index, :, :, :] = self.g_vect[index]
		pass


	def ask(self):
		"""
		:return:
			gradients for accelerations (shape: batch_size x 3 x h x w)
			hidden_states for optimizer
		"""
		self.indices = np.random.choice(self.dataset_size, self.batch_size)  # TODO: replace=False!

		with torch.enable_grad():
			# compute gradients wrt accelerations
			asked_grads = torch.zeros(self.batch_size, 3, self.h, self.w, device=device)
			if params.training and params.data.noise_acc > 0:
				# add noise to asked grads
				asked_grads += params.data.noise_acc * torch.randn_like(asked_grads)
			asked_grads.requires_grad_()

			loss_dict = loss(self.x[self.indices],
									   self.v[self.indices],
									   self.a[self.indices] + asked_grads,
									   self.a_exts[self.indices],
									   self.bc_masks[self.indices], self.bc_positions[self.indices],
									   self.M, self.stiffnesses[self.indices], self.shearings[self.indices],
									   self.bendings[self.indices])

			l = torch.sum(loss_dict["L"])  # input grads should be independent of batch size => use sum instead of mean
			l.backward()
			grads = asked_grads.grad

			if params.training and params.data.noise_grad > 0:
				# add noise to gradients
				grads = grads + params.data.noise_grad * torch.randn_like(grads)
			# print(f'loss: { l.item()}, self.a: {float(torch.norm(self.a[self.indices], p=2)):.3f}')
		return grads, [self.hidden_states[i] for i in self.indices]


	def ask_inference(self, retain_graph=True, frame_counter=None):
		if frame_counter:
			a_index = frame_counter
		else:
			a_index = self.indices[0]

		# reuse self._asked_grads for efficiency
		with torch.enable_grad():
			asked_grads = self._asked_grads
			# # asked_grads.grad = None  # reset gradients
			# # asked_grads.data.zero_()
			loss_dict = loss_efficient(self.x,
									   self.v,
									   self.a[a_index] + asked_grads,
									   self.a_exts,
									   self.bc_masks,
									   self.bc_positions,
									   self.M,
									   self.stiffnesses,
									   self.shearings,
									   self.bendings,
									   )


			l = loss_dict["L"].sum()
			grads = torch.autograd.grad(l, asked_grads,
										retain_graph=retain_graph,
										create_graph=retain_graph)[0]
		self._asked_grads = self._asked_grads.detach().requires_grad_(True)
		return grads, [self.hidden_states[a_index]]

	def ask_inference_jit(self, retain_graph=True, frame_counter=None):
		self.indices = np.zeros(1, dtype=int)
		if frame_counter:
			a_index = frame_counter
		else:
			a_index = self.indices[0]

		# reuse self._asked_grads for efficiency
		with torch.enable_grad():
			asked_grads = self._asked_grads
			asked_grads.grad = None  # reset gradients
			asked_grads.data.zero_()
			loss_dict = loss_efficient_jit(self.x,
									   self.v,
									   self.a[a_index] + asked_grads,
									   self.a_exts,
									   self.bc_masks,
									   self.bc_positions,
									   self.M,
									   self.stiffnesses,
									   self.shearings,
									   self.bendings,
									   k_repulsive=float(k_repulsive),
									   f_connectivity_edges=f_connectivity_edges,  # (E,2) long tensor, on same device
									   resolution_tpe=resolution_tpe
									   )

			l = loss_dict["L"].sum()
			grads = torch.autograd.grad(l, asked_grads,
										retain_graph=retain_graph,
										create_graph=retain_graph)[0]

		self._asked_grads = self._asked_grads.detach().requires_grad_(True)
		return grads, [self.hidden_states[i] for i in self.indices]


	def tell(self, step, hidden_states=None):
		"""
		:step: update step for accelerations for gradients given by ask
		:hidden_states: list of hidden states that are returned in following ask() calls => this is helpful to store the optimizer state
		:return: loss to optimize neural update-step-model (scalar values)
		"""
		hidden_states = [None for _ in self.indices] if hidden_states is None else hidden_states
		self.iterations[self.indices] = self.iterations[self.indices] + 1
		acc = self.a[self.indices] + step
		self.a[self.indices] = acc.detach()

		# compute loss => CODO: scaling of loss?
		loss_dict = loss(self.x[self.indices], self.v[self.indices], acc, self.a_exts[self.indices],
						self.bc_masks[self.indices], self.bc_positions[self.indices], self.M,
						self.stiffnesses[self.indices], self.shearings[self.indices], self.bendings[self.indices])

		l, E_int = loss_dict["L"], loss_dict["E_int"]

		# TODO: set bc?
		# update step if iterations_per_timestep is reached
		for i, index in enumerate(self.indices):
			self.hidden_states[index] = hidden_states[i]
			if self.iterations[index] % self.iterations_per_timestep == 0:
				self.T[index] = self.T[index] + dt
				self.update_env(index)
				if E_int[i] > 20000 and params.training:
					print(f'reset env:{index}, E_int: {E_int[i]:.2f}')
					self.reset_env(index)

		# reset environments eventually TODO: check that / reset environment, if E_int becomes too large!
		self.step += 1
		if self.step % (self.average_sequence_length * self.iterations_per_timestep / self.batch_size) == 0:  # ca x*batch_size steps until env gets reset => TODO attention!: average_sequence_length mut be divisible by (batch_size*iterations_per_timestep)!
			self.reset_env(int(self.reset_i))
			self.reset_i = (self.reset_i + 1) % self.dataset_size

		# wandb log
		if self.step % 100 == 0 and params.wandb.log:
			log_keys = ["L", "L_stiff", "L_shear", "L_bend", "L_ext", "L_inert", "M_stiff", "M_shear", "M_bend"] + (["L_repulsive", "M_repulsive"] if params.cloth.repulsive.k > 0 else [])
			log_dict = {k: loss_dict[k].mean() for k in log_keys}
			wandb.log(log_dict, step=self.step)

		return torch.mean(l)

	def tell_inference(self, step, hidden_states=None, detach_acc=False, bc_velocity=None, frame_counter=None):
		"""
		For inference
		"""
		hidden_states = [None] if hidden_states is None else hidden_states
		self.iterations[self.indices] += 1
		acc_index = self.indices if frame_counter is None else frame_counter
		acc = self.a[acc_index] + step
		if detach_acc:
			self.a[acc_index] = acc.detach()
		else:
			self.a[acc_index] = acc

		# TODO: set bc?
		self.hidden_states[0] = hidden_states[0]
		if self.iterations[0] % self.iterations_per_timestep == 0:
			self.T[0] = self.T[0] + dt
			self.update_env(0, bc_velocity, frame_counter=frame_counter)	 # first update state
			self.a.detach_()		

		# reset environments eventually TODO: check that / reset environment, if E_int becomes too large!
		self.step += 1
		# then detach a
		# self.a = self.a.detach()		# NOTE modification 07.03 added this line
		return None


