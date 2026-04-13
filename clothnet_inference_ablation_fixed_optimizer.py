"""
V3: add ground truth for MeshgraphnetRP
"""
import math
import os
import sys
import uuid
from pathlib import Path
import logging
import traceback

from fontTools.misc.psCharStrings import t1Operators
from matplotlib.colors import LightSource
from torch import optim

from compute_forces import compute_wind_force
from configs.config_common import motion_presets
from dataset_utils import DatasetToSingleChannel, generate_vertex_force
from generate_json_conf import setup_handle_traj, setup_handle_traj_gt, get_path_from_gt_input, \
    obj_name_to_handle_ind_list_dict, change_sequence_speed
from grid_mesh import GridMesh
from preprocess import transform_positions
from tri_to_quad_mesh import load_obj_with_uv, batch_trimesh_to_quadmesh, batch_trimesh_to_quadmesh_torch
from utils import generate_ffmpeg_cmd, get_unique_filename, get_face_areas_batch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Logger import Logger
from dataset_cloth3 import DatasetCloth, loss
from get_param2 import toCuda, get_hyperparam, params, device
from fnopt import get_Net3 as get_Net
from sft import evaluation
from sft.render import opencv_projection, ComputeViewMatrix, render_pytorch, render_single, \
    render_metamizer_video, render_metamizer_video_with_gt
from sft.utils import loadJson

import subprocess
import numpy as np
from pytorch3d.transforms import axis_angle_to_matrix
import time
from PIL import Image
import matplotlib.pyplot as plt
import torch
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras, PerspectiveCameras
from pytorch3d.io import load_obj, save_obj


class Rollout():
    def __init__(self, simulation_frames=500, evaluate=False, device='cuda'):
        self.simulation_frames = simulation_frames
        self.device = device
        self.dtype = params.net.dtype
        self.evaluate = evaluate

        self.opt_type = params.ablation.optimizer.opt_type  # "sgd" | "adam" | "gd"
        self.lr = params.ablation.optimizer.lr
        self.momentum = params.ablation.optimizer.momentum  # only for sgd
        self.beta1, self.beta2 = 0.9, 0.999  # for Adam

        # for LBFGS
        self.lbfgs_hist = params.ablation.optimizer.lbfgs_hist
        self.lbfgs_step = self.lr
        self.lbfgs_eps = params.ablation.optimizer.lbfgs_eps
        self.max_step_norm = params.ablation.optimizer.max_step_norm

        self.eps = 1e-8
        # {"v": vel, "m": m, "v_hat": v}
        self.opt_state = {}

        self.iter_to_record = [3, 5, 8, 10, 15, 20, 25, 30, 50, 100, 200, 500]
        self.loss_state = {
            key: [] for key in self.iter_to_record
        }
        self.losses = []
        pass

    def initializeParameters(self, scene, input_data=None):
        self.scene_parameters = scene
        if 'mesh_resolution' in scene:
            self.mesh_resolution = scene['mesh_resolution']
            self.h = self.mesh_resolution
            self.w = self.mesh_resolution
        else:
            self.h = scene['mesh_resolution_w'] # note inverse order because of dataset implementation
            self.w = scene['mesh_resolution_h'] # note inverse order because of dataset implementation
            self.mesh_resolution = f'{self.h}x{self.w}'
        if 'mesh_size_h' in scene and 'mesh_size_w' in scene:
            self.mesh_size_h = scene['mesh_size_h']
            self.mesh_size_w = scene['mesh_size_w']
        else:
            self.mesh_size_h = scene['mesh_size']
            self.mesh_size_w = scene['mesh_size']
        self.mesh_area = self.mesh_size_h * self.mesh_size_w
        self.real_scene = True if self.scene_parameters['scene'][0] == 'R' else False
        self.dir_dataset = scene['dataset_dir']
        self.input_data = input_data


        self.frame_counter = 0
        self.t_iter = 0
        self.epoch_counter = 0
        self.dpi = 200
        self.speed_factor = None

        self.time_conversion = 50  # 1s = 50 [NN-t]     # TODO Check this
        # self.length_conversion = (self.h - 1) / self.scene_parameters["mesh_size"]  # 1m = 31 [NN-m]

        # note in metamizer, length_conversion is set to the resolution of the tested cloth, instead of training resolution.????? -> May be wrong. Should set to training resolution.
        self.length_conversion = (self.h - 1) / self.mesh_size_w  # 1m = 31 [NN-m]
        # print('length_conversion:', self.length_conversion)

        self.bc_n_x = math.ceil(self.h / 32)

        self.unique_id = uuid.uuid4().hex[:8]
        self.dir_rgb = os.path.abspath(os.path.dirname(self.scene_parameters["result_chamfer_file"]) + '/rendered_rollout/tmp'+self.unique_id)
        if params.inference.rollout.save_render_metamizer and not os.path.exists(self.dir_rgb):
            os.makedirs(self.dir_rgb)

        self.dt = params.cloth.dt

        self.renderer = params.inference.renderer.lower()
        print('renderer:', self.renderer)

    def initializeMesh(self):
        verts, faces, aux = load_obj(str(Path(self.dir_dataset) / self.scene_parameters["mesh_file"]), load_textures=True, device=self.device)

        self.rest_positions = verts.to(dtype=self.dtype)
        self.faces = faces.verts_idx.to(dtype=torch.int32)
        self.faces_uv = faces.textures_idx.to(dtype=torch.int32)
        self.uv = aux.verts_uvs.to(dtype=torch.float32)

        if 'GINO' in params.net.name:
            u_range = torch.linspace(0, 1, self.h)
            v_range = torch.linspace(0, 1, self.w)
            uu, vv = torch.meshgrid(u_range, v_range, indexing='xy')
            uv_grid = torch.cat([uu.unsqueeze(0), vv.unsqueeze(0)]).permute(1, 2, 0)
            self.uv_gino = toCuda(uv_grid.flatten(0, 1))


        if "transform" in self.scene_parameters:
            if "rotate" in self.scene_parameters["transform"]:
                # apply axis-angle rotation
                rot = np.array(self.scene_parameters["transform"][
                                   "rotate"])  # rot is a (4, ) array for axis angle representation (angle, x, y, z)
                angle_rad = np.deg2rad(rot[0])
                axis = rot[1:] / np.linalg.norm(rot[1:])
                axis_angle = angle_rad * axis
                axis_angle_tensor = torch.tensor(axis_angle, dtype=self.dtype, device=self.device)
                rotation_matrix = axis_angle_to_matrix(axis_angle_tensor.unsqueeze(0)).squeeze(0)
                self.rest_positions = self.rest_positions @ rotation_matrix.T
            if "translate" in self.scene_parameters["transform"]:
                # apply translation
                translation = np.array(self.scene_parameters["transform"]["translate"])
                translation = torch.tensor(translation, dtype=self.dtype, device=self.device)
                self.rest_positions += translation

        self.rest_positions = self.rest_positions * self.length_conversion
        self.rest_positions = self.rest_positions.clone()
        self.original_uv = self.uv.clone()
        self.uv = self.uv.clone()

        self.positions_net = self.rest_positions.transpose(0, 1).view(3, self.h, self.w).type(self.dtype)
        velocities_net = torch.zeros(3, self.h, self.w, device=self.device)
        self.x_v = torch.cat([self.positions_net, velocities_net]).unsqueeze(0)

        if self.input_data and self.input_data['motion_code']:
            handle_ind_list = self.scene_parameters["handle_ind_list"]
            self.handle_mask = torch.ones_like(self.rest_positions).bool()
            for handle_ind in handle_ind_list:
                self.handle_mask[handle_ind, :] = False

            # load mgnrp mesh
            obj_path = str(Path(self.dir_dataset) / self.scene_parameters["mesh_file"])
            verts, _, _ = load_obj(obj_path, load_textures=True, device=self.device)

            if self.input_data['mode'] == 'arbitrary':
                handle_traj = setup_handle_traj(verts, self.input_data['motion_code'], list(reversed(handle_ind_list)),params.inference.rollout.framerate) # (F, V, 3)

            elif self.input_data['mode'] == 'with_gt':
                gt_path = get_path_from_gt_input(self.input_data, self.scene_parameters['root_mgnrp'])
                handle_traj_mgn = setup_handle_traj_gt(gt_path, obj_name_to_handle_ind_list_dict[self.input_data['obj_code']])
                handle_traj = torch.zeros((handle_traj_mgn.shape[0], self.h * self.w, 3), device=self.device, dtype=self.dtype)
                handle_traj[:, list(reversed(handle_ind_list))] = handle_traj_mgn[:, obj_name_to_handle_ind_list_dict[self.input_data['obj_code']]]
                #extend the first frame to match gt length
                handle_traj = torch.cat([handle_traj, handle_traj[-1:]], dim=0)

            if 'speed' in self.input_data and self.input_data['speed'] != 1.0:
                self.speed_factor = float(self.input_data['speed'])
                if self.input_data['mode'] == 'arbitrary':
                    hold_frames = 300
                else:
                    hold_frames = 0
                handle_traj = change_sequence_speed(handle_traj, self.speed_factor, hold_frames=hold_frames)


            handle_distance_mgnrp = (handle_traj[0, handle_ind_list[0]] - handle_traj[0, handle_ind_list[1]]).norm()
            handle_distance_ours = (self.rest_positions[handle_ind_list[0]] - self.rest_positions[handle_ind_list[1]]).norm()
            scale_ratio = handle_distance_ours / handle_distance_mgnrp
            handle_disp = handle_traj - torch.where(self.handle_mask, handle_traj, verts)
            self.R12 = torch.tensor([
                [1, 0, 0.],
                [0, 0, -1.],
                [0., 1., 0.]
            ], device=self.device).unsqueeze(0).repeat(handle_traj.shape[0], 1, 1)
            self.T12 = torch.tensor([0., 0., 0.], device=self.device).unsqueeze(0).repeat(handle_traj.shape[0], 1, 1)
            handle_disp = transform_positions(handle_disp.permute(0, 2, 1), self.R12, self.T12)
            self.handle_traj = torch.where(self.handle_mask, torch.zeros_like(self.rest_positions), self.rest_positions) + handle_disp * scale_ratio
            self.handle_traj = self.handle_traj.permute(0, 2, 1).reshape(self.handle_traj.shape[0], 3, self.h, self.w).to(self.device)


            self.handle_mask = self.handle_mask.permute(1, 0).reshape(3, self.h, self.w)
            self.simulation_frames = self.handle_traj.shape[0] - 1
            self.original_dataset = DatasetCloth(self.h, self.w, 1, 1,
                                                 params.inference.rollout.n_frames,
                                                 iterations_per_timestep=params.inference.iterations_per_timestep,
                                                 stiffness_range=params.cloth.stretching_range,
                                                 shearing_range=params.cloth.shearing_range,
                                                 bending_range=params.cloth.bending_range, a_ext_range=params.cloth.g)
            self.test_dataset = DatasetToSingleChannel(self.original_dataset)

            # compute bc velocity
            zero = torch.zeros_like(self.handle_traj[0:1])  # shape (1,3,32,32)
            diffs = self.handle_traj[1:] - self.handle_traj[:-1]  # shape (F-1,3,32,32)
            self.bc_vel_list = torch.cat([zero, diffs], dim=0)
            pass


    def initializeNetwork(self):
        network = toCuda(get_Net(params))
        logger = Logger(get_hyperparam(params), use_csv=False, use_tensorboard=False)

        print('load_date_time:', params.inference.load_date_time)
        if params.net.name == 'MeshGraphNets2':
            date_time, index = logger.load_state_mgn2(network, None, datetime=params.inference.load_date_time,
                                                      index=params.inference.load_index, device=self.device)

        else:
            date_time, index = logger.load_state(network, None, datetime=params.inference.load_date_time, index=params.inference.load_index, device=self.device)
        print(f"loaded: {date_time}, {index}")
        self.load_index = index
        self.cloth_net = network
        self.cloth_net.eval()

        # self.positions_net = self.rest_positions.transpose(0, 1).view(3, self.h, self.w).type(self.dtype)
        # velocities_net = torch.zeros(3, self.h, self.w, device=self.device)
        # self.x_v = torch.cat([self.positions_net, velocities_net]).unsqueeze(0)
        # self.original_dataset = DatasetCloth(self.h, self.w, 1, 1,
        #                                  params.inference.rollout.n_frames,
        #                                  iterations_per_timestep=params.inference.iterations_per_timestep,
        #                                  stiffness_range=params.cloth.stretching_range,
        #                                  shearing_range=params.cloth.shearing_range,
        #                                  bending_range=params.cloth.bending_range, a_ext_range=params.cloth.g)
        #
        #
        # self.test_dataset = DatasetToSingleChannel(self.original_dataset)

    def initializeCameraMatrix(self):
        self.crop_size = self.scene_parameters["upper_right_corner"] - self.scene_parameters["lower_left_corner"]
        if self.real_scene:
            self.initialize_real_camera()
        else:
            self.initialize_synthetic_camera()

    def initialize_real_camera(self):
        image_size = self.scene_parameters['image_size']
        self.image_size = (int(image_size[0]), int(image_size[1]))

        if self.renderer == 'nvdiffrast':
            self.projection = opencv_projection(
                image_size=self.image_size[::-1],
                optical_center=self.scene_parameters["optical_center"],
                focal_lengths=self.scene_parameters["focal_length"],
                z_near=0.01,
                z_far=1000.0,
            ).unsqueeze_(0).to(self.device)

            view_matrix = ComputeViewMatrix(
                self.scene_parameters["camera_position"],
                self.scene_parameters["camera_forward"],
                self.scene_parameters["camera_up"],
            ).T
            self.cameras = self.projection @ torch.from_numpy(view_matrix).to(self.device)
        elif self.renderer == 'pytorch3d':
            R = torch.tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]])
            focal_length = self.scene_parameters["focal_length"]
            optical_center = self.scene_parameters["optical_center"]
            self.cameras = PerspectiveCameras(focal_length=(focal_length,),
                                              principal_point=(optical_center,),
                                              in_ndc=False,
                                              image_size=(self.image_size,),
                                              R=R,
                                              device=self.device)

    def initialize_synthetic_camera(self):
        image_size = np.array([self.scene_parameters['height'], self.scene_parameters['width']])
        self.image_size = (int(image_size[0]), int(image_size[1]))

        height, width = self.image_size

        camera_pos = torch.tensor(self.scene_parameters["camera_position"],
                                  device=self.device, dtype=self.dtype)[None, :]
        up_dir = torch.tensor(self.scene_parameters["camera_up"],
                              device=self.device, dtype=self.dtype)[None, :]
        object_pos = torch.tensor(self.scene_parameters["object_position"],
                                  device=self.device, dtype=self.dtype)[None, :]

        R, T = look_at_view_transform(eye=camera_pos, at=object_pos, up=up_dir, device=self.device)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        if self.renderer == 'pytorch3d':
            self.cameras = cameras
        elif self.renderer == 'nvdiffrast':
            fov_degrees = cameras.fov[0].item()
            fov_radians = np.deg2rad(fov_degrees)

            # Compute focal lengths in pixels from FOV and image size
            fx = (0.5 * width) / np.tan(0.5 * fov_radians)
            fy = (0.5 * height) / np.tan(0.5 * fov_radians)
            focal_length = np.array([fx, fy])

            # Assume optical center is at the center of the image (can be adjusted if needed)
            cx = width / 2.0
            cy = height / 2.0
            optical_center = np.array([cx, cy])

            self.projection = opencv_projection(
                image_size=self.image_size[::-1],
                optical_center=optical_center,
                focal_lengths=focal_length,
                z_near=0.01,
                z_far=1000.0,
            ).unsqueeze_(0).to(self.device)

            camera_forward = self.scene_parameters['object_position'] - self.scene_parameters["camera_position"]
            camera_forward = camera_forward / np.linalg.norm(camera_forward)
            camera_up = self.scene_parameters["camera_up"]
            camera_up = camera_up - np.dot(camera_up, camera_forward) * camera_forward
            camera_up /= np.linalg.norm(camera_up)

            view_matrix = torch.from_numpy(ComputeViewMatrix(
                self.scene_parameters["camera_position"],
                camera_forward,
                camera_up,
            )).T.to(self.device)
            self.cameras = self.projection @ view_matrix

    def initializeOptimization(self):
        gravity = torch.tensor(self.scene_parameters["gravity"], device=self.device, dtype=torch.float32)

        # self.external_forces = torch.tensor([0, 0, -1]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)       # default
        # self.external_forces = torch.tensor([0, 0, -0.125]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)       # todo delete
        # todo check coefficient here -> according to loss(), we shouldn't modify the mass
        # mass_coeff = 32*32 / self.mesh_resolution**2
        # mass_coeff = 10

        self.external_forces = torch.tensor(gravity * self.length_conversion / (self.time_conversion * self.time_conversion), device=self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(2).unsqueeze(3).requires_grad_(True)   # todo check
        self.vertex_forces = torch.zeros((1, self.simulation_frames, 3, self.h, self.w), device=self.device, dtype=torch.float32)
        # add customed vertex forces
        # self.vertex_forces[0] = generate_vertex_force(self.h, self.w, self.simulation_frames, device=self.device,
        #                                               mode='multi_impact')

        self.predicted_a = torch.zeros((self.simulation_frames + 1, 3, self.h, self.w), device=self.device, dtype=torch.float32)
        self.predicted_pos = torch.zeros((self.simulation_frames + 1, self.h * self.w, 3), device=self.device, dtype=torch.float32)
        self.predicted_pos[0] = self.rest_positions / self.length_conversion
        self.wind_force = 0

        self.scales = []
        self.max_scales = []
        self.gradients = []

        self.stretching_stiffness = torch.tensor([params.inference.material.stretching], device=self.device, dtype=torch.float32)
        self.shearing_stiffness = torch.tensor([params.inference.material.shearing], device=self.device, dtype=torch.float32)
        self.bending_stiffness = torch.tensor([params.inference.material.bending], device=self.device, dtype=torch.float32)


        self.original_dataset.reset0_sft_env(self.positions_net)
        self.original_dataset.set_optimizable(self.external_forces, self.stretching_stiffness, self.shearing_stiffness, self.bending_stiffness)
        self.original_dataset.set_materials(self.stretching_stiffness, self.shearing_stiffness, self.bending_stiffness)
        # print('self.external_forces:', self.external_forces)
        if self.input_data and self.input_data['motion_code']:
            # set bc
            self.original_dataset.set_bc_positions(self.handle_traj[self.frame_counter])

    def initialize(self, scene, input_data=None):
        print("Start: Initialization")
        t_start = time.perf_counter()

        self.initializeParameters(scene, input_data)
        self.initializeMesh()
        self.initializeNetwork()
        self.initializeCameraMatrix()
        self.initializeOptimization()

        # texture
        texture_path = str(Path(self.dir_dataset) / self.scene_parameters["texture_file"])
        texture = np.array(Image.open(texture_path)) / 255.0        # TODO resize?
        self.texture_image = torch.tensor(texture)[None, ...].to(dtype=torch.float32).to(self.device)

        if self.evaluate:
            ground_truth_point_clouds, point_clouds_lengths = evaluation.loadGroundTruthMGNRP(
                self.scene_parameters, self.input_data, device=self.device
            )
            # our_point_clouds = torch.zeros_like(ground_truth_point_clouds)
            self.point_clouds = {"ground_truth": ground_truth_point_clouds,
                                 # "ours": our_point_clouds,
                                 "lengths": point_clouds_lengths}

        self.cloth_m = np.load(
            os.path.join(os.path.dirname(self.scene_parameters['mesh_file_mgn']), 'square_1024', 'cloth_m.npy'))
        self.mass_conversion = np.sum(self.cloth_m) / (self.h * self.w)  # average mass per vertex
        t_end = time.perf_counter()
        print(f"Done:  Initialization in {t_end - t_start:.3f} s\n")


    def step(self):
        a_ext = self.external_forces + self.vertex_forces[:, self.frame_counter]
        # wind
        if params.inference.rollout.wind_density > 0.0:
            a_ext = a_ext + self.wind_force
        last_iter = ((self.t_iter + 1) % params.inference.iterations_per_timestep == 0)

        self.original_dataset.set_optimizable(a_ext, self.stretching_stiffness, self.shearing_stiffness, self.bending_stiffness)
        # t0 = time.perf_counter()
        # grads, hidden_states = self.test_dataset.ask()  # get gradients

        grads, hidden_states = self.test_dataset.ask_inference(retain_graph=False)  # get gradients
        # t1 = time.perf_counter()

        # ② 根据 opt_type 计算 update_steps ---------------------
        key = "default"  # 你这里只有 batch=1，可用固定 key
        if key not in self.opt_state:
            self.opt_state[key] = {
                "v": torch.zeros_like(grads),  # SGD momentum
                "m": torch.zeros_like(grads),  # Adam m
                "v_hat": torch.zeros_like(grads),  # Adam v
                "lbfgs": {
                    "s_list": [],                 # list[Tensor], 最近 m 次的 s_k（上一步更新）
                    "y_list": [],                 # list[Tensor], 最近 m 次的 y_k（梯度差）
                    "prev_grad": None,            # 上一步梯度
                    "prev_update": None,          # 上一步已应用的 update_steps
                    "m": getattr(self, "lbfgs_hist", 10)  # 历史长度（默认10）
                }
            }

        opt = self.opt_state[key]
        new_hidden_states = None
        if self.opt_type == "sgd":
            # --- SGD with momentum
            opt["v"] = self.momentum * opt["v"] - self.lr * grads
            update_steps = opt["v"]

        elif self.opt_type == "adam":
            # --- Adam update
            opt["m"] = self.beta1 * opt["m"] + (1 - self.beta1) * grads
            opt["v_hat"] = self.beta2 * opt["v_hat"] + (1 - self.beta2) * grads.pow(2)
            m_hat = opt["m"] / (1 - self.beta1)
            v_hat = opt["v_hat"] / (1 - self.beta2)
            update_steps = -self.lr * m_hat / (v_hat.sqrt() + self.eps)

        elif self.opt_type == "lbfgs":
            state = opt["lbfgs"]

            # ---- (A) 先用上一步的信息更新 (s, y) 历史 -----------------
            # 进入本次迭代时，grads = g_k；若有上一轮信息，就构造 y_{k-1}、s_{k-1}
            if state["prev_grad"] is not None and state["prev_update"] is not None:
                y = grads - state["prev_grad"]  # y_{k-1} = g_k - g_{k-1}
                s = state["prev_update"]  # s_{k-1} = x_k - x_{k-1}（这里就是上一步的更新）
                ys = torch.sum(y * s)

                # 只在曲率充分（y^T s > 0）时保留该对，避免数值不稳
                if torch.isfinite(ys) and ys.item() > self.lbfgs_eps:
                    if len(state["s_list"]) >= state["m"]:
                        state["s_list"].pop(0)
                        state["y_list"].pop(0)
                    # 注意：直接保存 Tensor 引用即可；若担心后续被原地改动，可 .detach().clone()
                    state["s_list"].append(s.detach())
                    state["y_list"].append(y.detach())

            # ---- (B) 两环递推，得到近似的 -H_k g_k 方向 ------------------
            # q <- g_k
            q = grads.detach().clone()
            alphas = []

            # 第一环：从近到远
            for s_i, y_i in zip(reversed(state["s_list"]), reversed(state["y_list"])):
                rho_i = 1.0 / (torch.sum(y_i * s_i) + self.lbfgs_eps)
                alpha_i = rho_i * torch.sum(s_i * q)
                q = q - alpha_i * y_i
                alphas.append(alpha_i)

            # 初始尺度 H0：用最后一对 (s,y) 的 Barzilai-Borwein 比例，若无历史则用 1
            if len(state["s_list"]) > 0:
                s_last, y_last = state["s_list"][-1], state["y_list"][-1]
                gamma = torch.sum(s_last * y_last) / (torch.sum(y_last * y_last) + self.lbfgs_eps)
            else:
                gamma = torch.tensor(1.0, device=grads.device, dtype=grads.dtype)

            r = gamma * q

            # 第二环：从远到近
            for i in range(len(state["s_list"])):
                s_i, y_i = state["s_list"][i], state["y_list"][i]
                rho_i = 1.0 / (torch.sum(y_i * s_i) + self.lbfgs_eps)
                beta_i = rho_i * torch.sum(y_i * r)
                alpha_i = alphas[-(i + 1)]  # 与第一环对齐
                r = r + s_i * (alpha_i - beta_i)

            direction = -r

            # ---- (C) 步长与可选信任域 -------------------------------
            step_size = self.lbfgs_step  # 常数步长，保证“等梯度评估”公平
            update_steps = step_size * direction

            if self.max_step_norm is not None:
                # 可选：裁剪到信任域，避免过大步导致不稳
                step_norm = update_steps.norm()
                if torch.isfinite(step_norm) and step_norm.item() > self.max_step_norm:
                    update_steps = update_steps * (self.max_step_norm / (step_norm + 1e-12))

            # ---- (D) 记录当前梯度与更新，供下一轮构造 (s,y) -------------
            state["prev_grad"] = grads.detach()
            state["prev_update"] = update_steps.detach()

            # 若这是一个时间步内的最后一次内迭代，重置历史，避免跨时间步污染
            if last_iter:
                state["s_list"].clear()
                state["y_list"].clear()
                state["prev_grad"] = None
                state["prev_update"] = None

        elif self.opt_type == "gd":  # "gd"
            update_steps = -self.lr * grads

        elif self.opt_type == "ours":
            update_steps, new_hidden_states = self.cloth_net(grads, hidden_states)


        bc_vel = None
        if self.input_data and self.input_data['motion_code']:
            bc_vel = self.bc_vel_list[self.frame_counter]
            self.original_dataset.set_bc_positions(self.handle_traj[self.frame_counter])

            i_iter_in_step = self.t_iter % params.inference.iterations_per_timestep

            # if i_iter_in_step + 1 in self.iter_to_record:
            #     loss_dict = loss(self.original_dataset.x[self.original_dataset.indices],
            #                      self.original_dataset.v[self.original_dataset.indices], self.original_dataset.a,
            #                      self.original_dataset.a_exts[self.original_dataset.indices],
            #                      self.original_dataset.bc_masks[self.original_dataset.indices],
            #                      self.original_dataset.bc_positions[self.original_dataset.indices],
            #                      self.original_dataset.M,
            #                      self.original_dataset.stiffnesses[self.original_dataset.indices],
            #                      self.original_dataset.shearings[self.original_dataset.indices],
            #                      self.original_dataset.bendings[self.original_dataset.indices],
            #                      self.original_dataset.E_repulsive[self.original_dataset.indices])
            #
            #     self.loss_state[i_iter_in_step + 1].append(loss_dict['L'].item())
                # print('iter:', i_iter_in_step + 1, 'L_cloth:', loss_dict['L'].item(), 'E_int:', loss_dict['E_int'].item(),)

            if last_iter:
                loss_dict = loss(self.original_dataset.x[self.original_dataset.indices],
                                 self.original_dataset.v[self.original_dataset.indices], self.original_dataset.a,
                                 self.original_dataset.a_exts[self.original_dataset.indices],
                                 self.original_dataset.bc_masks[self.original_dataset.indices],
                                 self.original_dataset.bc_positions[self.original_dataset.indices],
                                 self.original_dataset.M,
                                 self.original_dataset.stiffnesses[self.original_dataset.indices],
                                 self.original_dataset.shearings[self.original_dataset.indices],
                                 self.original_dataset.bendings[self.original_dataset.indices],
                                 self.original_dataset.E_repulsive[self.original_dataset.indices])
                self.losses.append(loss_dict["L"].item())

        _ = self.test_dataset.tell_inference(update_steps, new_hidden_states, detach_acc=True, bc_velocity=bc_vel)



        if last_iter:
            self.gradients.append(torch.norm(grads, p=2).item())

            # compute cloth loss


        if last_iter:
            # update wind force
            cloth_v = self.original_dataset.v.permute(0, 2, 3, 1).reshape(-1, 3) * self.time_conversion / self.length_conversion  # convert to m/s
            cloth_f_area = get_face_areas_batch(vertices=self.original_dataset.x.permute(0, 2, 3, 1).reshape(1, -1, 3), faces=self.faces) / (self.length_conversion ** 2)  # compute face area
            cloth_pos = self.original_dataset.x.permute(0, 2, 3, 1).reshape(-1, 3) / self.length_conversion  # convert to m
            face_tensor = self.faces

            wind_force = compute_wind_force(self.original_dataset.M.squeeze().reshape(-1, 1),
                                            cloth_f_area.permute(1, 0),
                                            face_tensor,
                                            cloth_v,
                                            cloth_pos,
                                            wind_density=params.inference.rollout.wind_density)

            # TODO check if convert or not
            wind_force = wind_force * self.length_conversion / (self.time_conversion * self.time_conversion * self.mass_conversion)
            self.wind_force = wind_force.reshape(1, self.h, self.w, 3).permute(0, 3, 1, 2)

        if last_iter:  # visualize only at a new timestep (a timestep can take several iterations to optimize)
            self.frame_counter += 1
            if self.frame_counter % 50 == 0:
                # compute l_cloth avg and std for each recorded iteration
                # l_cloth_avg = [np.mean(self.loss_state[iter]) for iter in self.iter_to_record]
                # l_cloth_std = [np.std(self.loss_state[iter]) for iter in self.iter_to_record]
                # for i, iter in enumerate(self.iter_to_record):
                #     if iter <= params.inference.iterations_per_timestep:
                #         print(f"iter {iter}: L_cloth = {l_cloth_avg[i]:.3f} ± {l_cloth_std[i]:.3f}")

                l_cloth_avg = np.mean(self.losses)
                l_cloth_std = np.std(self.losses)
                print(f"L_cloth = {l_cloth_avg:.3f} ± {l_cloth_std:.3f}")
                print(
                    f'current frame: {self.frame_counter:>4} | '
                    f'input grads: {self.gradients[-1]:.3f} | '
                    f'update step: {float(torch.norm(update_steps, p=2)):.3f}'
                )
            index = 0
            x = self.original_dataset.x[index]
            predicted_pos = x.view(3, -1).transpose(0, 1) / self.length_conversion
            self.predicted_pos[self.frame_counter] = predicted_pos.squeeze()

            if params.inference.visualize_3d:  # visualize 3D cloth
                print('!!!!!!!!!!!!!!!!! visualize 3D !!!!!!!!!!!!!!!!!!!')
                x_np = x.cpu().numpy()
                bc_masks = self.original_dataset.bc_masks[index, 0].cpu()
                ls = LightSource(azdeg=315, altdeg=45)  # Control the direction of the light
                rgb = ls.shade(x_np[2], cmap=plt.cm.viridis, vert_exag=0.1, blend_mode='soft')
                plt.figure(1, figsize=(800 / self.dpi, 800 / self.dpi), dpi=self.dpi)
                plt.clf()
                fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"}, num=1, computed_zorder=False)
                surf = ax.plot_surface(x_np[0], x_np[1], x_np[2], linewidth=0.1, antialiased=False, zorder=4, rstride=1,
                                       cstride=1)  # ,alpha=0.5) # cloth surface
                # boundary conditions
                cond = (bc_masks > 0).nonzero()
                ax.scatter(x_np[0, cond[:, 0], cond[:, 1]], x_np[1, cond[:, 0], cond[:, 1]], x_np[2, cond[:, 0], cond[:, 1]],
                           marker='o', color='g', depthshade=False, zorder=5)  # boundarys conditions
                ax.grid(False)
                ax.set_axis_off()  # Completely removes the 3D box

                # Remove the axes (pane color)
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False

                # Hide the axes lines
                ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # X-axis
                ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Y-axis
                ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Z-axis

                # Optionally, hide the ticks
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])

                """
                q_stride, q_l=8,10
                # gradients
                ax.quiver(x[0,::q_stride,::q_stride], x[1,::q_stride,::q_stride], x[2,::q_stride,::q_stride], \
                    q_l*grads[0,0,::q_stride,::q_stride], q_l*grads[0,1,::q_stride,::q_stride], q_l*grads[0,2,::q_stride,::q_stride],color='r')

                # accelerations
                ax.quiver(x[0,::q_stride,::q_stride], x[1,::q_stride,::q_stride], x[2,::q_stride,::q_stride], \
                    q_l*a[0,::q_stride,::q_stride], q_l*a[1,::q_stride,::q_stride], q_l*a[2,::q_stride,::q_stride],color='g')
                """

                ax.set_zlim(-2 * params.inference.height * 0.6, 1.01)
                ax.set_xlim(-params.inference.height * 0.6, params.inference.height * 0.6)
                ax.set_ylim(-params.inference.height * 0.6, params.inference.height * 0.6)
                plt.title(f"timestep: {self.original_dataset.T[index].cpu().numpy()[0]}")


                self.path_metamizer = f"plots/{get_hyperparam(params).replace(' ', '_').replace(';', '_')}/cloth/{params.inference.load_date_time}/stiff_{params.inference.material.stretching} shear_{params.inference.material.shearing} bend_{params.inference.material.bending} iters_{params.inference.iterations_per_timestep}/tmp{self.unique_id}"
                os.makedirs(self.path_metamizer, exist_ok=True)
                assert os.path.exists(self.path_metamizer), f"Failed to create path: {self.path_metamizer}"
                # plt.savefig(f"{self.path_metamizer}/frame_{str(self.frame_counter).zfill(4)}.png", dpi=self.dpi)

                plt.draw()
                plt.pause(0.01)

                if params.inference.save_obj and self.frame_counter % 20 == 0:  # save mesh every 10 frames
                    # save mesh
                    mesh_path = f"{self.path_metamizer}/frame_{str(self.frame_counter).zfill(4)}.obj"
                    print('mesh_path:', mesh_path)
                    save_obj(mesh_path, x.permute(1, 2, 0).reshape(-1, 3) / self.length_conversion, self.faces)

            if params.inference.visualize_scaling:  # visualize, how scaling changes during update steps
                plt.figure(2)
                plt.clf()
                stride = 1  # len(scales)//200+1
                plt.semilogy(self.scales[::stride])
                plt.xlabel("iteration")
                plt.ylabel("scale")
                plt.legend(["scales"])
                plt.title(f"Scale of Metamizer, {params.inference.iterations_per_timestep} iterations per timestep, {self.h} x {self.w}")

                plt.draw()
                plt.pause(0.01)

            if params.inference.visualize_grads:  # visualize, how norm of loss gradients changes during update steps
                print('!!!!!!!!!!!!!!!!! visualize gradients !!!!!!!!!!!!!!!!!!!')
                plt.figure(3, figsize=(1600 / self.dpi, 800 / self.dpi), dpi=self.dpi)
                plt.clf()
                stride = 1  # len(scales)//200+1
                plt.semilogy(self.gradients[::stride])
                plt.xlabel("iteration")
                plt.ylabel("gradient norm")
                plt.title(
                    f"Gradient Norm reduction of Metamizer, {params.inference.iterations_per_timestep} iterations per timestep, {self.h} x {self.w}")
                plt.draw()
                plt.pause(0.01)

        self.t_iter += 1

    def save_acc_visu(self):
        a_all = self.predicted_a.cpu().detach().numpy()
        a_all = np.clip(a_all, np.percentile(a_all, 5), np.percentile(a_all, 95))
        vmin_global = np.min(a_all)
        vmax_global = np.max(a_all)
        self.dir_acc = os.path.abspath(os.path.dirname(self.scene_parameters["result_chamfer_file"]) + '/rendered_acc/tmp' + self.unique_id)
        if not os.path.exists(self.dir_acc):
            os.makedirs(self.dir_acc)
        for self.frame_counter in range(a_all.shape[0]):
            a = a_all[self.frame_counter].squeeze()
            plt.imshow(a[0].transpose(), vmin=vmin_global, vmax=vmax_global)
            plt.colorbar()
            save_path_acc = os.path.join(self.dir_acc, str(self.frame_counter).zfill(4) + ".png")
            print('save path:', save_path_acc)
            plt.savefig(save_path_acc, dpi=100)
            plt.close()


def main():
    simulation_frames = params.inference.rollout.n_frames
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scene_list = params.inference.rollout.json
    motion_code_list = motion_presets[params.inference.rollout.input_data.motion_code] if params.inference.rollout.using_handle_traj else [None]
    print('params.inference.rollout.input_data.motion_code:', params.inference.rollout.input_data.motion_code)
    print('params.inference.rollout.using_handle_traj', params.inference.rollout.using_handle_traj)
    print('motion_code_list:', motion_code_list)
    task_list = [(file_name, motion_code) for file_name in scene_list for motion_code in motion_code_list]
    print('\n\n\n**************************')
    print('device:', device)
    print('model:', params.net.name)
    print('renderer:', params.inference.renderer.lower())
    print('N_iter_per_step:', params.inference.iterations_per_timestep)
    print('dtype:', params.net.dtype)
    if params.inference.rollout.using_handle_traj:
        print('motion codes:', motion_code_list)
        print('mode:', params.inference.rollout.input_data.mode)
    print('**************************\n\n\n')

    for file_name, motion in task_list:
        save_npy = params.inference.rollout.save_npy
        print(f'file_name: {file_name}, motion: {motion}' )
        scene = loadJson(file_name)
        input_data = dict(params.inference.rollout.input_data)
        input_data['motion_code'] = motion
        evaluate = (params.inference.rollout.evaluate and input_data is not None and input_data['mode'] == 'with_gt')
        print('input_data:', input_data)
        opt = Rollout(simulation_frames=simulation_frames, evaluate=evaluate, device=device)
        if params.net.name != 'MGNRP':
            opt.initialize(scene=scene, input_data=input_data)
        else:
            opt.initializeParameters(scene, input_data)
            opt.initializeMesh()
            opt.initializeOptimization()

        Y = params.inference.material.stretching
        S = params.inference.material.shearing
        B = params.inference.material.bending
        WD = params.inference.rollout.wind_density
        iter = params.inference.iterations_per_timestep
        epoch = opt.load_index if hasattr(opt, 'load_index') else 800   # 800 is MGNRP default epoch
        res = opt.mesh_resolution if 'mesh_resolution' in opt.__dict__ else scene['mesh_resolution']

        mode = input_data['mode']
        save_dir = f"evaluation/S_MGNRP/npy_results/{params.net.name}{params.inference.postfix}/{mode}/"
        if params.net.name == 'MGNRP':
            # npy_file_name = f"{input_data['obj_code']}_{res}_{input_data['motion_code']}_e_{epoch}.npy"
            result_file_bone_name = f"template_mgnrp_quad_{res}_{input_data['motion_code']}_e_{epoch}_abl_optimizer_{opt.opt_type}_lr_{opt.lr}"

        else:
            result_file_bone_name = f"{input_data['obj_code']}_{input_data['motion_code']}_e_{epoch}_Y{Y}_S{S}_B{B}_WD{WD}_iter{iter}_res{res}_abl_optimizer_{opt.opt_type}_lr_{opt.lr}"

            if opt.speed_factor is not None:
                result_file_bone_name += f'_speed{opt.speed_factor}'
            if 'bc_desc' in opt.scene_parameters:
                result_file_bone_name += f"_bc{opt.scene_parameters['bc_desc']}"
        # if params.net.name == 'MGNRP':
        #     # npy_file_name = f"{input_data['obj_code']}_{res}_{input_data['motion_code']}_e_{epoch}.npy"
        #     npy_file_name = f"template_mgnrp_quad_{res}_{input_data['motion_code']}_e_{epoch}.npy"
        # else:
        #     npy_file_name = f"{input_data['obj_code']}_{input_data['motion_code']}_e_{epoch}_Y{Y}_S{S}_B{B}_WD{WD}_iter{iter}_res{res}.npy"
        npy_file_name = result_file_bone_name + '.npy'
        # if opt.speed_factor is not None:
        #     npy_file_name = npy_file_name[:-4] + f'_speed{opt.speed_factor}.npy'  # add speed to filename if provided

        save_path_npy = os.path.join(save_dir, npy_file_name)
        print(f'npy file path: {os.path.abspath(save_path_npy)}')

        if evaluate:
            log_path = Path(f"evaluation/S_MGNRP/eval_log_{params.net.name}{opt.mesh_resolution}_abl_optimizer_{opt.opt_type}_lr_{opt.lr}.txt")
            print(f'logging save to: {os.path.abspath(log_path)}')
            log_path.parent.mkdir(parents=True, exist_ok=True)
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)s | %(message)s",
                handlers=[
                    logging.FileHandler(log_path, mode='a'),  # append to log file
                    logging.StreamHandler()  # print to console
                ]
            )
            logging.getLogger('matplotlib').setLevel(logging.WARNING)
            # logging.info('-' * 80)
        do_rollout = save_npy or not params.inference.rollout.using_handle_traj
        if motion and save_npy:
            os.makedirs(save_dir, exist_ok=True)
            if os.path.exists(save_path_npy):
                print(f"[Skip] Already exists: {os.path.abspath(save_path_npy)}")
                do_rollout = False
                save_npy = False

            else:
                try:
                    # Create a lock file to prevent concurrent prediction
                    lock_path = save_path_npy + '.lock'
                    lock_dir = os.path.dirname(lock_path)
                    os.makedirs(lock_dir, exist_ok=True)
                    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.close(fd)
                except FileExistsError:
                    print(f"[Skip] Another process is predicting (lock present): {lock_path}")
                    do_rollout = False
                    save_npy = False


        from torch.profiler import profile, record_function, ProfilerActivity

        time1 = time.perf_counter()

        per_frame_duration = 0.0
        if do_rollout:
            try:
                while (opt.t_iter < opt.simulation_frames * params.inference.iterations_per_timestep):
                    opt.step()

                # with profile(
                #         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                #         record_shapes=True,
                #         with_stack=True
                # ) as prof:
                #     while (opt.t_iter < opt.simulation_frames * params.inference.iterations_per_timestep):
                #         opt.step()
                #         if opt.t_iter == 100:
                #             break
                # # 按 CUDA 时间排序，列出前 10 个 kernel
                # print(prof.key_averages().table(
                #     sort_by="cuda_time_total", row_limit=10
                # ))
                # sys.exit()
            except Exception as e:
                logging.error(
                    f"ERROR during rollout | "
                    f"Model: {params.net.name}{params.inference.postfix} | "
                    f"Iter/ts: {params.inference.iterations_per_timestep:<3} | "
                    f"Motion: {str(motion):<12} | "
                    f"Y:" f"{params.inference.material.stretching:6} | "
                    f"S:" f"{params.inference.material.shearing:<4} | "
                    f"B:" f"{params.inference.material.bending:<6} | "
                    f"Wind: {params.inference.rollout.wind_density:<3} | "
                    # f"Res: {getattr(opt.mesh_resolution):<3} | "
                    f"{type(e).__name__}: {str(e)}"
                )
                traceback.print_exc()
                continue
            finally:
                os.remove(lock_path)

            print("------+--------+-----------+-------------------------+-------------------------------------------------------------------------------------")
            time2 = time.perf_counter()
            duration = time2 - time1
            per_frame_duration = duration / opt.simulation_frames
            print(f"Done in {duration: .4f} s, per frame: {per_frame_duration:.4f}s")

        if motion and save_npy:
            R_inv = torch.linalg.inv(opt.R12)
            T_inv = -torch.matmul(opt.T12, R_inv.transpose(-1, -2))

            predicted_pos = transform_positions(opt.predicted_pos.permute(0, 2, 1), R_inv, T_inv)
            save_dict = {
                'position': predicted_pos.cpu().numpy(),  # [T, V, 3]
                'face': opt.faces.cpu().numpy(),  # [F, 3]
                'input_data': input_data,
                'model_name': f"{params.net.name}{params.inference.postfix}"
            }
            print(f"Saving rollout results to {os.path.abspath(save_path_npy)}")
            np.save(save_path_npy, save_dict)

        try:
            npy_results = np.load(save_path_npy, allow_pickle=True).item()
        except FileNotFoundError:
            print(f"File not found: {os.path.abspath(save_path_npy)}. Skipping this sequence.")
            continue
        else:
            print(f"Loaded results from {os.path.abspath(save_path_npy)}")


        gt_x = None
        predicted_pos = torch.from_numpy(npy_results['position']).to(device)
        if predicted_pos.shape[0] != opt.R12.shape[0]:
            predicted_pos = predicted_pos[1:]
        if opt.evaluate:
            # load gt mesh
            gt_path = get_path_from_gt_input(input_data, scene['root_mgnrp'])
            gt_x = torch.from_numpy(np.load(os.path.join(gt_path, "cloth_pos.npy"))).to(device)
            # gt_x = transform_positions(gt_x.permute(0, 2, 1), opt.R12, opt.T12).cpu().numpy()
            _, gt_f, gt_uv = load_obj_with_uv(scene['mesh_file_mgn'])

            # predicted_pos = torch.from_numpy(npy_results['position']).to(device)
            # if predicted_pos.shape[0] != opt.R12.shape[0]:
            #     predicted_pos = predicted_pos[1:]
            # predicted_pos = transform_positions(predicted_pos.permute(0, 2, 1), opt.R12, opt.T12).cpu().numpy()

            ground_truth_point_clouds = opt.point_clouds["ground_truth"]
            our_point_clouds = torch.zeros_like(ground_truth_point_clouds)
            # positions = torch.from_numpy(npy_results['position']).to(device)
            faces = torch.from_numpy(npy_results['face']).to(device)

            assert predicted_pos.shape[0] == ground_truth_point_clouds.shape[0], \
                f"Mismatch in number of frames: {predicted_pos.shape[0]} vs {ground_truth_point_clouds.shape[0]}"
            try:

                print('Evaluating e3D...')
                gt_f = torch.from_numpy(gt_f).to(device)
                gt_uv = torch.from_numpy(gt_uv).to(device)
                quad_verts, quad_quads = batch_trimesh_to_quadmesh_torch(gt_x, gt_f, gt_uv, (opt.h, opt.w))


                # compute e3D
                # quad_verts, quad_quads = batch_trimesh_to_quadmesh(
                #     gt_x.cpu().numpy(), gt_f, gt_uv,
                #     resolution=(opt.h, opt.w)
                # )
                quad_verts = quad_verts.reshape(quad_verts.shape[0], -1, 3)
                assert quad_verts.shape == predicted_pos.shape, \
                    f"Mismatch in number of frames: {quad_verts.shape} vs {predicted_pos.shape}"
                e3d = evaluation.computeMeshDistance(quad_verts, predicted_pos, 0, len(gt_x))

                # compute chamfer distance
                for i in range(ground_truth_point_clouds.shape[0]):
                    pos = predicted_pos[i]
                    if pos.numel() == 0 or pos.shape[0] == 0:
                        raise AssertionError(f"[Frame {i}] predicted_pos is empty.")

                    if not torch.isfinite(pos).all():
                        raise AssertionError(f"[Frame {i}] NaN or Inf detected in predicted_pos.")

                    if torch.any(pos.isnan()):
                        raise AssertionError(f"[Frame {i}] NaN in predicted_pos.")

                    if torch.any(pos.abs() > 1e8):
                        raise AssertionError(f"[Frame {i}] Unreasonably large value in predicted_pos.")

                    if torch.all(pos == 0):
                        raise AssertionError(f"[Frame {i}] All-zero predicted_pos.")

                    if faces.max() >= pos.shape[0]:
                        raise AssertionError(
                            f"[Frame {i}] Invalid face index: max face index {faces.max().item()} >= {pos.shape[0]}"
                        )

                    our_point_clouds[i] = evaluation.sampleMesh(opt.point_clouds["lengths"][i],
                                                                predicted_pos[i],
                                                                faces,
                                                                device=device)
                print('Evaluating Chamfer distance...')
                chamfer_distance = evaluation.computeChamferDistance(ground_truth_point_clouds,
                                                                     our_point_clouds,
                                                                     0,
                                                                     len(opt.point_clouds["ground_truth"]),
                                                                     opt.point_clouds["lengths"],
                                                                     inverse=False,
                                                                     scale=1)
            except Exception as e:
                logging.error(f"Error computing Chamfer distance: {e}")
                chamfer_distance = float('nan')
                traceback.print_exc()


            logging.info(
                f"Model: {params.net.name}{params.inference.postfix} | "
                f"Iter/ts: {params.inference.iterations_per_timestep:<3} | "
                f"Motion: {motion:<12} | "
                f"Y:" f"{params.inference.material.stretching:6} | "
                f"S:" f"{params.inference.material.shearing:<4} | "
                f"B:" f"{params.inference.material.bending:<6} | "
                f"Wind: {params.inference.rollout.wind_density:<3} | "                
                f"t/frame: {per_frame_duration:<8.3e} | "
                f"Chamfer: {chamfer_distance:.4f} | "
                f"e3d: {e3d:.4f} | "
            )


        mp4_file = result_file_bone_name + '.mp4'
            # mp4_file = f"{input_data['obj_code']}_{input_data['motion_code']}_e_{epoch}_Y{Y}_S{S}_B{B}_WD{WD}_iter{iter}_res{res}.mp4"

        # if opt.speed_factor is not None:
        #     mp4_file = mp4_file[:-4] + f'_speed{opt.speed_factor}.mp4'  # add speed to filename if provided

        if params.inference.rollout.save_render_metamizer:
            seq_name = scene["scene"]
            if motion:
                save_dir = f'evaluation/{seq_name}/rendered_rollout/{params.net.name}{params.inference.postfix}/{input_data["mode"]}'
            else:
                save_dir = f'evaluation/{seq_name}/rendered_rollout/{params.net.name}{params.inference.postfix}'
            save_path = os.path.join(save_dir, mp4_file)
            gt_alpha = 0
            pred_alpha = 1

            skip = False
            if os.path.exists(save_path):
                skip = True
                print(f"[Skip] Already exists: {os.path.abspath(save_path)}")

            else:
                try:
                    # Create a lock file to prevent concurrent rendering
                    lock_path = save_path + '.lock'
                    lock_dir = os.path.dirname(lock_path)
                    os.makedirs(lock_dir, exist_ok=True)
                    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.close(fd)
                except FileExistsError:
                    print(f"[Skip] Another process is rendering (lock present): {lock_path}")
                    skip = True

            if not skip:
                try:
                    # predicted_pos = torch.from_numpy(npy_results['position']).to(device)
                    # if predicted_pos.shape[0] != opt.R12.shape[0]:
                    #     predicted_pos = predicted_pos[1:]

                    predicted_pos = transform_positions(predicted_pos.permute(0, 2, 1), opt.R12, opt.T12).cpu().numpy()
                    x_list = predicted_pos.transpose(0, 2, 1).reshape(predicted_pos.shape[0], 3, opt.h, opt.w)# list of (3, H, W)
                    bc_mask_list = np.repeat(opt.original_dataset.bc_masks[0, 0].cpu().numpy()[np.newaxis, ...], len(x_list), axis=0)

                    print(f"Saving rendering to: {os.path.abspath(save_path)}")
                    plot_error_map = True
                    if mode == 'arbitrary':
                        gt_x = None
                        gt_f = None
                        gt_uv = None
                        plot_error_map = False
                        pass
                    elif mode == 'with_gt':
                        if opt.h != opt.w:
                            # cases where view range doesn't follow gt mesh
                            gt_alpha = 0
                            gt_x = None
                            gt_f = None
                            gt_uv = None
                            plot_error_map = False
                        else:
                            if opt.speed_factor is not None or opt.h != opt.w or 'bc_desc' in opt.scene_parameters:
                                # cases where gt doesn't need to be rendered
                                gt_alpha = 0
                                plot_error_map = False
                            gt_path = get_path_from_gt_input(input_data, scene['root_mgnrp'])
                            if gt_x is None and opt.speed_factor is None:
                                gt_x = torch.from_numpy(np.load(os.path.join(gt_path, "cloth_pos.npy"))).to(device)
                            R12 = opt.R12[0].unsqueeze(0).repeat(gt_x.shape[0], 1, 1)
                            T12 = opt.T12[0].unsqueeze(0).repeat(gt_x.shape[0], 1, 1)
                            gt_x = transform_positions(gt_x.permute(0, 2, 1), R12, T12)
                            _, gt_f, gt_uv = load_obj_with_uv(scene['mesh_file_mgn'])

                            gt_uv = torch.from_numpy(gt_uv).to(device)
                            gt_f = torch.from_numpy(gt_f).to(device)

                    render_metamizer_video_with_gt(
                        x_preds=torch.from_numpy(x_list).to(device),  # (F,3,H,W)
                        gt_vertices_list=gt_x,  # (F,V,3)
                        gt_faces=gt_f,  # (n_faces,3)
                        gt_uv=gt_uv,
                        bc_mask_list=bc_mask_list,
                        save_path=save_path,
                        fps=params.inference.rollout.framerate,
                        gt_alpha=gt_alpha,         # 0.4
                        pred_alpha=pred_alpha,
                        azim=-60,       #-60
                        elev=30,        #30
                        dpi=600,            # 800
                        debug=params.inference.visualize_3d,
                        show_handle_points=params.inference.rollout.show_handle_points,
                        show_error_map=plot_error_map,

                        # xlim=(0, 1),
                        # ylim=(-1, 1),
                        # zlim=(0, 1)
                    )
                    print(f"Render video saved to: {os.path.abspath(save_path)}")
                finally:
                    os.remove(lock_path)

        print('motion:', motion)
        print('save_render_3d:', params.inference.rollout.save_render_3d)
        if motion and params.inference.rollout.save_render_3d:
            viewport_dict = {'def': (-60, 30), 'side': (-90, 90), 'front': (0, 0)}
            face_info = GridMesh(height=opt.h, width=opt.w).generate_triangles().numpy()
            viewport = 'def'
            fps = params.inference.rollout.framerate
            render_dir = os.path.abspath(f'evaluation/S_MGNRP/renders/{params.net.name}{params.inference.postfix}/{input_data["mode"]}')

            if not os.path.exists(render_dir):
                os.makedirs(render_dir)

            result_path = os.path.join(render_dir, mp4_file)

            if os.path.exists(result_path):
                print(f"[Skip] Render file already exists: {result_path}")
            else:
                print(f'saving 3d render to: {result_path}')
                predicted_pos = npy_results['position']
                if mode == 'arbitrary':
                    position_list = [[predicted_pos]]
                    face_info_list = [[face_info]]
                elif mode == 'with_gt':
                    gt_path = get_path_from_gt_input(input_data, scene['root_mgnrp'])
                    gt_x = np.load(os.path.join(gt_path, "cloth_pos.npy"))
                    _, f, _ = load_obj_with_uv(scene['mesh_file_mgn'])
                    position_list = [[predicted_pos, gt_x]]
                    face_info_list = [[face_info, f]]
                try:
                    render_single(position_list, face_info_list, viewport_dict[viewport], result_path, fps, debug=params.inference.visualize_3d)
                except Exception as e:
                    print(f"Error during 3D rendering: {e}")
                    traceback.print_exc()

                else:
                    print('3D render saved at:', result_path)

        if do_rollout and params.inference.visualize_scaling:  # visualize, how scaling changes during update steps
            dpi = 200
            plt.figure(2)
            plt.clf()
            stride = 1  # len(scales)//200+1
            plt.semilogy(opt.scales[::stride])
            plt.xlabel("iteration")
            plt.ylabel("scale")
            plt.legend(["scales"])
            plt.draw()
            plt.savefig(f"{render_dir}/V_SCALE_{params.net.name}{params.inference.postfix}_RES{opt.mesh_resolution}_Y{params.inference.material.stretching}_S{params.inference.material.shearing}_B{params.inference.material.bending}_EP{opt.load_index}_iters{params.inference.iterations_per_timestep}.png", dpi=dpi)
        if do_rollout and params.inference.visualize_grads:
            dpi = 200
            plt.figure(3, figsize=(1600 / dpi, 800 / dpi), dpi=dpi)
            plt.clf()
            stride = 1  # len(scales)//200+1
            plt.semilogy(opt.gradients[::stride])
            plt.xlabel("iteration")
            plt.ylabel("gradient norm")
            plt.title(f"Gradient Norm reduction of Metamizer, {params.inference.iterations_per_timestep} iterations per timestep, {opt.h} x {opt.w}")
            plt.draw()
            plt.savefig(f"{render_dir}/V_GRADS_{params.net.name}{params.inference.postfix}_RES{opt.mesh_resolution}_Y{params.inference.material.stretching}_S{params.inference.material.shearing}_B{params.inference.material.bending}_EP{opt.load_index}_iters{params.inference.iterations_per_timestep}.png", dpi=dpi)
            pass

if __name__ == '__main__':
    params.wandb.log = False    # disable wandb logging
    params.training = False
    torch.set_num_threads(10)
    print('device:', device)
    print("MKLDNN enabled:", torch.backends.mkldnn.enabled)
    print("MKL enabled:", torch.backends.mkl.is_available())
    print("torch.get_num_threads():", torch.get_num_threads())
    print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
    main()