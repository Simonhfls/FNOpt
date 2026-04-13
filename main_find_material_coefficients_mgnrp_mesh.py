# run this file from the "Code" directory using
# python main_sft.py --config_file=SMP_param_a_gated3a.yaml --config_name=default
import logging
import math
import random
import subprocess
import sys
import os
import uuid
from pathlib import Path
from matplotlib import pyplot as plt
from pytorch3d.transforms import axis_angle_to_matrix

from compute_forces import compute_wind_force
from configs.config_common import motion_presets
from generate_json_conf import setup_handle_traj, get_path_from_gt_input, setup_handle_traj_gt, \
    obj_name_to_handle_ind_list_dict
from preprocess import transform_positions
from tri_to_quad_mesh import load_obj_with_uv, batch_trimesh_to_quadmesh, batch_trimesh_to_quadmesh_torch, \
    save_quadmesh_obj_with_uv, generate_uv_grid
from utils import get_face_areas_batch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Logger import Logger
from dataset_cloth3 import DatasetCloth
from dataset_utils import DatasetToSingleChannel
from get_param2 import params, device
from metamizer import get_Net3 as get_Net
from sft import evaluation
from sft.utils import loadJson
import numpy as np
import time
from PIL import Image
from get_param2 import toCuda, get_hyperparam
import os
import torch
from pytorch3d.io import load_obj, save_obj

torch.autograd.set_detect_anomaly(False)

class Optimization():
    def __init__(self, save_render=False, save_mesh=False, debug=False, seq_desc='default_seq', device='cuda'):
        self.save_render = save_render
        self.save_mesh = save_mesh
        self.debug = debug
        self.device = device
        self.dtype = params.net.dtype
        self.seq_desc = seq_desc
        # self.f_start = 0
        # self.f_end = 60

    
    def initializeParameters(self, scene, input_data=None):
        print('initialize parameters...')
        self.scene_parameters = scene
        self.mesh_resolution = 32 if 'mesh_resolution' not in scene else scene['mesh_resolution']
        self.real_scene = True if self.scene_parameters['scene'][0] == 'R' else False
        self.h = self.mesh_resolution # 32
        self.w = self.mesh_resolution # 32
        self.frame_counter = 0
        self.t_iter = 0
        # self.frames_per_epoch = min(3, self.scene_parameters["n_images"])
        # self.frames_per_epoch = min(49, self.scene_parameters["n_images"])
        self.frames_per_epoch = 60
        self.new_frame_period = params.inference.sft.new_frame_period
        self.epoch_counter = 0
        self.simulation_frames = 60
        self.start_frame = 0
        self.dpi = 200
        self.time_conversion = 50  # 1s = 50 [NN-t]
        # note in metamizer, length_conversion is set to the resolution of the tested cloth, instead of training resolution.
        self.length_conversion = (self.mesh_resolution - 1) / self.scene_parameters["mesh_size"]  # 1m = 31 [NN-m]
        self.bc_n_x = math.ceil(self.mesh_resolution / 32)
        self.dir_dataset = scene['dataset_dir']
        self.input_data = input_data
        self.unique_id = uuid.uuid4().hex[:8]

        if self.save_render:
            self.dir_render = os.path.abspath(os.path.dirname(self.scene_parameters["result_chamfer_file"]) + '/rendered/' + self.seq_desc)
            self.dir_rgb = os.path.join(self.dir_render, 'rgb')
            self.dir_acc = os.path.join(self.dir_render, 'acc')
            print('render dir:', self.dir_render)
            if not os.path.exists(self.dir_rgb):
                os.makedirs(self.dir_rgb)
            if not os.path.exists(self.dir_acc):
                os.makedirs(self.dir_acc)
        if self.debug:
            self.dir_debug = os.path.abspath(os.path.dirname(self.scene_parameters["result_chamfer_file"]) + '/debug/' + self.seq_desc + '/' + self.unique_id)
            if not os.path.exists(self.dir_debug):
                os.makedirs(self.dir_debug)
            print('debug dir:', self.dir_debug)
        if self.save_mesh:
            self.dir_obj = os.path.dirname(os.path.dirname(self.scene_parameters['result_point_cloud_files'])) + '/obj/'
            if not os.path.exists(self.dir_obj):
                os.makedirs(self.dir_obj)
        self.renderer = params.inference.renderer.lower()



    def initializeMesh(self):
        print('initialize mesh...')
        verts, faces, aux = load_obj(str(Path(self.dir_dataset) / self.scene_parameters["mesh_file"]), load_textures=True, device=self.device)
        self.rest_positions = verts.to(dtype=torch.float32)
        self.faces = faces.verts_idx.to(dtype=torch.int32)
        self.faces_uv = faces.textures_idx.to(dtype=torch.int32)
        self.uv = aux.verts_uvs.to(dtype=torch.float32)

        if "transform" in self.scene_parameters:
            if "rotate" in self.scene_parameters["transform"]:
                # apply axis-angle rotation
                rot = np.array(self.scene_parameters["transform"]["rotate"])  # rot is a (4, ) array for axis angle representation (angle, x, y, z)
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

        self.positions = self.rest_positions.clone()
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

            gt_path = get_path_from_gt_input(self.input_data, self.scene_parameters['root_mgnrp'])
            handle_traj_mgn = setup_handle_traj_gt(gt_path, obj_name_to_handle_ind_list_dict[self.input_data['obj_code']])
            handle_traj = torch.zeros((handle_traj_mgn.shape[0], self.h * self.w, 3), device=self.device, dtype=self.dtype)
            handle_traj[:, list(reversed(handle_ind_list))] = handle_traj_mgn[:, obj_name_to_handle_ind_list_dict[self.input_data['obj_code']]]
            #extend the first frame to match gt length
            handle_traj = torch.cat([handle_traj, handle_traj[-1:]], dim=0)

            # load gt
            # self.gt_x = torch.from_numpy(np.load(os.path.join(gt_path, "cloth_pos.npy"))).to(self.device)

            # Y = params.inference.material.stretching
            # S = params.inference.material.shearing
            # B = params.inference.material.bending
            Y = 1000
            S = 8
            B = 0.5
            WD = params.inference.rollout.wind_density
            iter = params.inference.iterations_per_timestep
            # epoch = self.load_index
            res = self.mesh_resolution
            npy_file_name = f"{self.input_data['obj_code']}_{self.input_data['motion_code']}_e_{100}_Y{Y}_S{S}_B{B}_WD{WD}_iter{iter}_res{res}.npy"
            mode = self.input_data['mode']
            save_dir = f"evaluation/S_MGNRP/npy_results/{params.net.name}{params.inference.postfix}/{mode}/"
            npy_file_path = os.path.join(save_dir, npy_file_name)
            self.gt_x = torch.from_numpy(np.load(npy_file_path, allow_pickle=True).item()['position']).to(self.device)

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
            self.R21 = torch.linalg.inv(self.R12)
            self.T21 = -torch.matmul(self.T12, self.R21.transpose(-1, -2))
            handle_disp = transform_positions(handle_disp.permute(0, 2, 1), self.R12, self.T12)
            self.gt_x = transform_positions(self.gt_x.permute(0, 2, 1), self.R12, self.T12)  # TODO check!!!!!!!

            self.handle_traj = torch.where(self.handle_mask, torch.zeros_like(self.rest_positions), self.rest_positions) + handle_disp * scale_ratio
            self.handle_traj = self.handle_traj.permute(0, 2, 1).reshape(self.handle_traj.shape[0], 3, self.h, self.w).to(self.device)


            self.handle_mask = self.handle_mask.permute(1, 0).reshape(3, self.h, self.w)
            # self.simulation_frames = self.handle_traj.shape[0] - 1
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

            _, gt_f, gt_uv = load_obj_with_uv(self.scene_parameters['mesh_file_mgn'])
            self.gt_f = torch.from_numpy(gt_f).to(self.device)
            self.gt_uv = torch.from_numpy(gt_uv).to(self.device)
            pass


    def initializeNetwork(self):
        print('initialize network...')

        network = toCuda(get_Net(params))
        logger = Logger(get_hyperparam(params), use_csv=False, use_tensorboard=False)
        print('load_date_time:', params.inference.load_date_time)
        date_time, index = logger.load_state(network, None, datetime=params.inference.load_date_time, index=params.inference.load_index, device=self.device)
        print(f"loaded: {date_time}, {index}")
        self.load_index = index
        self.cloth_net = network

        self.cloth_net.eval()
        for param in self.cloth_net.parameters():
            param.requires_grad = False
        pass


    def initializeOptimization(self):
        print('initialize optimization...')
        self.vertex_forces = torch.zeros((1, self.simulation_frames, 3, self.h, self.w), device=self.device, dtype=torch.float32)
        self.stretching_stiffness = torch.tensor([params.inference.material.stretching], device=self.device, dtype=torch.float32, requires_grad=True)
        self.shearing_stiffness = torch.tensor([params.inference.material.shearing], device=self.device, dtype=torch.float32, requires_grad=True)
        self.bending_stiffness = torch.tensor([params.inference.material.bending], device=self.device, dtype=torch.float32, requires_grad=True)
        self.original_dataset = DatasetCloth(self.h, self.w, 1, 1,
                                             1000,
                                             iterations_per_timestep=params.inference.sft.iterations_per_timestep,
                                             stiffness_range=params.cloth.stretching_range,
                                             shearing_range=params.cloth.shearing_range,
                                             bending_range=params.cloth.bending_range, a_ext_range=params.cloth.g)

        positions_net = self.rest_positions.transpose(0, 1).view(3, self.h, self.w).type(torch.float32)
        self.x = positions_net.clone().unsqueeze(0)

        bc_indices = None
        if "handles" in self.scene_parameters:
            bc_indices = self.scene_parameters["handles"]
        self.original_dataset.reset0_sft_env(self.x, bc_indices=bc_indices)        # todo check, no need ?
        self.test_dataset = DatasetToSingleChannel(self.original_dataset)

        gravity = torch.tensor(self.scene_parameters["gravity"], device=self.device, dtype=torch.float32)
        self.external_forces = torch.tensor(gravity * self.length_conversion / (self.time_conversion * self.time_conversion), device=self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(2).unsqueeze(3)

        print('initial external forces:', self.external_forces[0, 0, 0, 0].item(), self.external_forces[0, 1, 0, 0].item(),
              self.external_forces[0, 2, 0, 0].item())
        a_ext = self.external_forces + self.vertex_forces[:, self.frame_counter]

        self.parameters = [
            self.stretching_stiffness,
            self.shearing_stiffness,
            self.bending_stiffness,
        ]
        learning_rates = [
            params.inference.sft.lr.stretching,
            params.inference.sft.lr.shearing,
            params.inference.sft.lr.bending,
        ]
        self.optimizer = [None] * len(self.parameters)
        for i in range(len(self.parameters)):
            self.optimizer[i] = torch.optim.Adam([self.parameters[i]], lr=learning_rates[i]) # default
            # self.optimizer[i] = torch.optim.SGD([self.parameters[i]], lr=learning_rates[i], momentum=0)

        self.loss = torch.tensor([0.], dtype=torch.float32).to(self.device)
        self.predicted_a = torch.zeros((self.simulation_frames + 1, 3, self.h, self.w), device=self.device, dtype=torch.float32)
        self.scales = []
        self.original_dataset.reset0_sft_env(self.positions_net)
        self.original_dataset.set_optimizable(a_ext, self.stretching_stiffness, self.shearing_stiffness, self.bending_stiffness)
        self.original_dataset.set_bc_positions(self.handle_traj[self.frame_counter])
        self.wind_force = 0


    def printQuantities(self):
        t = time.perf_counter()
        if self.epoch_counter % 1 == 0:
            print(f"{self.epoch_counter:5d} | {t - self.time:6.2f} s | {t - self.time_start:7.2f} s | "
                  # f"Loss: {self.loss.item() * self.frames_per_epoch:.2e} {self.loss.item():.2e} | "
                  f"Loss: {self.loss.item() * self.frames_per_epoch:.2e} {self.loss.item():.2e} | "
                  f"{self.stretching_stiffness.item():.3e} "
                  f"{self.shearing_stiffness.item():.3e} "
                  f"{self.bending_stiffness.item():.3e} "
                  f"{(self.external_forces[0, 0, 0, 0].item() / self.length_conversion * self.time_conversion * self.time_conversion): .2e} "
                  f"{(self.external_forces[0, 1, 0, 0].item() / self.length_conversion * self.time_conversion * self.time_conversion): .2e} "
                  f"{(self.external_forces[0, 2, 0, 0].item() / self.length_conversion * self.time_conversion * self.time_conversion): .2e}   "
                  f"{self.chamfer_distance.item():.2e}")
        self.time = t

    def resetState(self):
        positions_net = self.rest_positions.transpose(0, 1).view(3, self.h, self.w).type(torch.float32)
        with torch.no_grad():
            self.original_dataset.reset_sft_env(0)
        self.original_dataset.set_position(positions_net)
        with torch.no_grad():
            self.positions[:] = self.rest_positions / self.length_conversion
            self.loss = torch.tensor([0.], dtype=torch.float32, device=self.device)
            self.predictions = []
            self.point_clouds["ours"] = torch.zeros_like(self.point_clouds["ground_truth"])
            self.point_clouds["ours"][0] = self.rest_positions / self.length_conversion
            self.wind_force = 0
            self.original_dataset.set_bc_positions(self.handle_traj[self.frame_counter])



    def clampOptimization(self):
        with torch.no_grad():
            self.stretching_stiffness[0] = max(10, self.stretching_stiffness[0])
            self.shearing_stiffness[0] = max(1e-2, self.shearing_stiffness[0])
            self.bending_stiffness[0] = max(1e-5, self.bending_stiffness[0])


    def initialize(self, scene, max_epochs, input_data=None):
        print("Start: Initialization")
        t_start = time.perf_counter()
        self.initializeParameters(scene, input_data)
        self.initializeMesh()
        self.initializeNetwork()
        self.initializeOptimization()
        if self.renderer == "nvdiffrast":
            self.context = dr.RasterizeCudaContext()

        # texture
        print('load ground truth...', end='')
        print('done')

        self.chamfer_distance = torch.tensor([0.0])
        self.chamfer_distances_epochs = torch.zeros((max_epochs), device=self.device)

        our_v = torch.zeros_like(self.gt_x)

        # convert gt_x to quad mesh TODO add it back
        # self.gt_x, self.f_quad = batch_trimesh_to_quadmesh_torch(self.gt_x, self.gt_f, self.gt_uv, (self.h, self.w))
        _, self.f_quad = batch_trimesh_to_quadmesh_torch(self.gt_x, self.gt_f, self.gt_uv, (self.h, self.w))

        self.gt_x = self.gt_x.reshape(self.gt_x.shape[0], -1, 3)
        our_v[0] = self.rest_positions / self.length_conversion
        self.point_clouds = {"ground_truth": self.gt_x, "ours": our_v}

        # debug save mesh (well aligned)
        save_obj('ours.obj', our_v[0], self.faces)
        save_obj('gt.obj', self.gt_x[0], self.faces)

        t_end = time.perf_counter()

        self.time = time.perf_counter()
        self.time_start = time.perf_counter()
        self.predictions = []

        self.cloth_m = np.load(
            os.path.join(os.path.dirname(self.scene_parameters['mesh_file_mgn']), 'square_1024', 'cloth_m.npy'))
        self.mass_conversion = np.sum(self.cloth_m) / (self.h * self.w)  # average mass per vertex
        print(f"Done:  Initialization in {t_end - t_start:.3f} s\n")
        print("Epoch |  Time   |  Total t  | Loss:   Full  per Frame |  Stretch    Shear     Bend     Wind x    Wind y    Wind z    |  e3D     ")
        print("------+---------+-----------+-------------------------+-------------------------------------------------------------------------")

    def step(self):
        # update vertex force
        a_ext = self.external_forces + self.vertex_forces[:, self.frame_counter]
        if params.inference.rollout.wind_density > 0.0:
            a_ext = a_ext + self.wind_force
        self.original_dataset.set_optimizable(a_ext, self.stretching_stiffness, self.shearing_stiffness, self.bending_stiffness)

        # print('current material:', self.original_dataset.stiffnesses, self.original_dataset.bendings, self.original_dataset.shearings)
        last_iter = ((self.t_iter + 1) % params.inference.sft.iterations_per_timestep == 0)
        grads, hidden_states = self.test_dataset.ask_sft(retain_graph=last_iter)    # TODO set true
        # print('mean_grad:', grads.mean().item())


        update_steps, new_hidden_states = self.cloth_net(grads, hidden_states)
        bc_vel = self.bc_vel_list[self.frame_counter]
        self.original_dataset.set_bc_positions(self.handle_traj[self.frame_counter])

        _ = self.test_dataset.tell_sft(update_steps, new_hidden_states, bc_velocity=bc_vel)

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
            index = 0
            x = self.original_dataset.x[index]  # FIXME no grads?
            x_pred = x.view(3, -1).transpose(0, 1) / self.length_conversion
            self.point_clouds["ours"][self.frame_counter] = x_pred
            self.predictions.append(x_pred)

            if self.frame_counter == self.frames_per_epoch:
                self.chamfer_distance = torch.tensor([0.0])
                loss_e3d = evaluation.computeMeshDistance(self.point_clouds["ground_truth"], self.point_clouds["ours"], self.start_frame, self.frame_counter)
                # loss_e3d = evaluation.computeMeshDistanceWithoutNormalization(self.point_clouds["ground_truth"], self.point_clouds["ours"], 0, self.frame_counter + 1)

                # print('loss_e3d:', loss_e3d.item(), 'loss_e3d_2:', loss_e3d_2.item())
                # mean over all frames
                self.loss += loss_e3d

                self.chamfer_distance = loss_e3d.detach()
                self.chamfer_distances_epochs[self.epoch_counter] = self.chamfer_distance
                self.printQuantities()

                # NOTE modification2: I deleted loss normalization, seems improving the results.
                # loss_norm = self.loss.detach() + 1e-3
                # self.loss = self.loss / loss_norm
                self.loss.backward()

                for i in range(len(self.parameters)):
                    # print grads of parameters
                    self.optimizer[i].step()
                    self.optimizer[i].zero_grad()

                if self.debug and self.epoch_counter % 20 == 0:
                    # save mesh
                    uv_grid = generate_uv_grid(32, 32, inv_u=True, inv_v=True, flatten=False)
                    for i in range(0, self.frames_per_epoch, 5):
                        obj_path = Path(self.dir_debug) / f'e{self.epoch_counter}_f{i}.obj'
                        save_quadmesh_obj_with_uv(obj_path, self.point_clouds["ours"][i].reshape(self.h, self.w, 3), self.f_quad, uv_grid)


                self.frame_counter = 0
                self.epoch_counter += 1

                self.resetState()
                self.clampOptimization()

                # successively add frames
                # print('frame_per_epoch:', self.frames_per_epoch, 'simulation_frames:', self.simulation_frames, 'epoch_counter:', self.epoch_counter, 'new_frame_period:', self.new_frame_period)
                if self.frames_per_epoch < self.simulation_frames and self.epoch_counter % self.new_frame_period == 0:
                    print('here add new frame')
                    self.frames_per_epoch += 1
                    if self.frames_per_epoch == self.simulation_frames:
                        print("Reached max frames")


        self.t_iter += 1


def main():
    # set seed
    torch.manual_seed(params.data.seed)
    np.random.seed(params.data.seed)
    random.seed(params.data.seed)
    scene_list = params.inference.rollout.json
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('\n\n\n**************************')
    print("Seed: ", params.data.seed)
    print('device:', device)
    print('model:', params.net.name)
    print('N_iter_per_step:', params.inference.sft.iterations_per_timestep)
    print('dtype:', params.net.dtype)
    print('save render:', params.inference.sft.save_render)
    print('debug:', params.inference.sft.debug)
    print('**************************\n\n\n')

    motion_code_list = motion_presets[params.inference.rollout.input_data.motion_code] if params.inference.rollout.using_handle_traj else [None]
    print('params.inference.rollout.input_data.motion_code:', params.inference.rollout.input_data.motion_code)
    print('motion_code_list:', motion_code_list)
    task_list = [(file_name, motion_code) for file_name in scene_list for motion_code in motion_code_list]

    for file_name, motion in task_list:
        input_data = dict(params.inference.rollout.input_data)
        input_data['motion_code'] = motion

        scene = loadJson(file_name)
        result_path = Path(scene['result_point_cloud_files'])
        log_path = result_path.parent.parent / f'results_{params.net.name}{params.inference.postfix}.txt'
        print(f'Logging save to: {log_path.resolve()}')
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(log_path, mode='a'),
                logging.StreamHandler()
            ]
        )
        seq_desc = f"{params.net.name}_{scene['mesh_resolution']}" if 'mesh_resolution' in scene else f"{params.net.name}_32"
        time1 = time.perf_counter()
        opt = Optimization(save_render=params.inference.sft.save_render, save_mesh=True, debug=params.inference.sft.debug, seq_desc=seq_desc, device=device)
        max_epochs = params.inference.sft.n_epochs_opt + 1
        opt.initialize(scene=scene, max_epochs=max_epochs, input_data=input_data)
        while (opt.epoch_counter < max_epochs):
            opt.step()

        print("------+---------+-----------+-------------------------+-------------------------------------------------------------------------------------")
        opt.printQuantities()
        logging.info(
            f"Model:         {params.net.name}{params.inference.postfix}\n"
            f"Iters:         {params.inference.sft.iterations_per_timestep:<3}     "
            f"n_epochs:      {params.inference.sft.n_epochs_opt:<4}     "
            f"lr_Y:          {params.inference.sft.lr.stretching:<8.1e} "
            f"lr_S:          {params.inference.sft.lr.shearing:<8.1e} "
            f"lr_B:          {params.inference.sft.lr.bending:<8.1e}\n"
            f"optimized_Y:   {opt.stretching_stiffness:<8.1e}\n"
            f"optimized_S:   {opt.shearing_stiffness:<8.1e}\n"
            f"optimized_B:   {opt.bending_stiffness:<8.1e}\n"
            f"Chamfer (e3d): {opt.chamfer_distance.item():.2e}"
        )

        torch.save(opt.chamfer_distances_epochs, scene["result_chamfer_file"])

        time2 = time.perf_counter()
        print(f"Done in {(time2 - time1): .2f} s")

        # TODO create a function for generating video

        if params.inference.sft.save_render:
            print(f"Generating accuracy heatmap video for resolution {opt.mesh_resolution}")
            acc_render_dir = opt.dir_acc
            output_file = f"ACC_{params.net}{params.inference.postfix}_{opt.mesh_resolution}_ep{opt.load_index}.mp4"

            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-framerate", f"{params.inference.sft.framerate}",
                "-pattern_type", "glob",
                "-i", f"{acc_render_dir}/*.png",
                "-frames:v", str(opt.simulation_frames),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                os.path.join(acc_render_dir, output_file)
            ]
            try:
                start_time = time.perf_counter()
                subprocess.run(ffmpeg_cmd, check=True)
                end_time = time.perf_counter()
                print(f"Accuracy heatmap video generation completed in {(end_time - start_time):.2f} seconds")
            except subprocess.CalledProcessError as e:
                print(f"Error generating accuracy heatmap video: {e}")


if __name__ == '__main__':
    params.wandb.log = False    # disable wandb logging
    params.training = False
    print('device:', device)
    print('params:', params)

    # print configuration
    print("********************")
    print("learning rates:")
    print(f"\tY: {params.inference.sft.lr.stretching}, "
          f"S: {params.inference.sft.lr.shearing}, "
          f"B: {params.inference.sft.lr.bending}, "
          f"external: {params.inference.sft.lr.external}, "
          f"vertex: {params.inference.sft.lr.vertex}, "
          f"uv: {params.inference.sft.lr.uv}")
    print('\nloss coefficients:')
    print(f"\trgb: {params.inference.sft.lc.rgb}, "
          f"silhouette: {params.inference.sft.lc.sil}, "
          f"uv_smooth: {params.inference.sft.lc.uv_smooth}, "
          f"vertex_shift: {params.inference.sft.lc.shift}")
    print("********************")

    if params.inference.renderer.lower() == "nvdiffrast":
        import nvdiffrast.torch as dr
    main()
