# run this file from the "Code" directory using
# python main.py --config_file=SMP_param_a_gated3a.yaml --config_name=default

import math
import random
import subprocess
import sys
import os

from matplotlib.colors import LightSource

from Logger import Logger
from dataset_cloth2 import DatasetCloth
from dataset_utils import DatasetToSingleChannel
from get_param2 import params, device
from metamizer import get_Net2 as get_Net
from sft import evaluation
from sft.render import opencv_projection, ComputeViewMatrix, render_pytorch, render_nvdiffrast
from sft.utils import update_obj_vertices, loadJson, grid_to_trimesh_faces

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from matplotlib import pyplot as plt
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
import numpy as np
from ema_pytorch import EMA
from pytorch3d.transforms import axis_angle_to_matrix
from torchvision.transforms.functional import gaussian_blur
import time
from PIL import Image
from get_param2 import toCuda, get_hyperparam
import os
import torch
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PerspectiveCameras, )
from pytorch3d.io import load_obj, save_obj



torch.autograd.set_detect_anomaly(False)


def test_grad_tensor(x):
    print('is_leaf:', x.is_leaf, '| requires_grad:', x.requires_grad, ', grad:',
          x.grad if x.grad is not None else 'None')


class Optimization():
    def __init__(self, save_render=False, save_mesh=False, debug=False, device='cuda'):
        self.save_render = save_render
        self.save_mesh = save_mesh
        self.debug = debug
        self.device = device
    
    def initializeParameters(self, scene, evaluate):
        self.scene_parameters = scene
        self.mesh_resolution = scene['mesh_resolution']
        self.real_scene = True if self.scene_parameters['scene'][0] == 'R' else False
        self.h = self.mesh_resolution # 32
        self.w = self.mesh_resolution # 32
        self.frame_counter = 0
        self.t_iter = 0
        # self.frames_per_epoch = min(3, self.scene_parameters["n_images"])
        self.frames_per_epoch = min(10, self.scene_parameters["n_images"])
        self.new_frame_period = params.inference.sft.new_frame_period
        self.epoch_counter = 0
        self.simulation_frames = self.scene_parameters["n_images"]
        self.evaluate = evaluate
        self.dpi = 200
        self.time_conversion = 1 / 0.02  # 1s = 50 [NN-t]
        # note in metamizer, length_conversion is set to the resolution of the tested cloth, instead of training resolution.
        self.length_conversion = (self.mesh_resolution - 1) / self.scene_parameters["mesh_size"]  # 1m = 31 [NN-m]
        self.bc_n_x = math.ceil(self.mesh_resolution / 32)
        if self.save_render:
            self.dir_render = os.path.abspath(os.path.dirname(self.scene_parameters["result_chamfer_file"]) + '/rendered/')
            self.dir_rgb = os.path.join(self.dir_render, 'rgb/')
            self.dir_acc = os.path.join(self.dir_render, 'acc/')
            if not os.path.exists(self.dir_rgb):
                os.makedirs(self.dir_rgb)
            if not os.path.exists(self.dir_acc):
                os.makedirs(self.dir_acc)
        if self.debug:
            self.dir_debug = os.path.abspath(os.path.dirname(self.scene_parameters["result_chamfer_file"]) + '/debug/')
            if not os.path.exists(self.dir_debug):
                os.makedirs(self.dir_debug)
        if self.save_mesh:
            self.dir_obj = os.path.dirname(os.path.dirname(self.scene_parameters['result_point_cloud_files'])) + '/obj/'
            if not os.path.exists(self.dir_obj):
                os.makedirs(self.dir_obj)
        self.renderer = params.inference.renderer.lower()
        print('renderer:', self.renderer)


    def initializeMesh(self):
        verts, faces, aux = load_obj(self.scene_parameters["mesh_file"], load_textures=True, device=self.device)
        self.rest_positions = verts.to(dtype=torch.float32)
        self.faces = faces.verts_idx.to(dtype=torch.int32)
        self.faces_uv = faces.textures_idx.to(dtype=torch.int32)
        self.uv = aux.verts_uvs.to(dtype=torch.float32)
        if "transform" in self.scene_parameters:
            if "rotate" in self.scene_parameters["transform"]:
                # apply axis-angle rotation
                rot = np.array(self.scene_parameters["transform"][
                                   "rotate"])  # rot is a (4, ) array for axis angle representation (angle, x, y, z)
                angle_rad = np.deg2rad(rot[0])
                axis = rot[1:] / np.linalg.norm(rot[1:])
                axis_angle = angle_rad * axis
                axis_angle_tensor = torch.tensor(axis_angle, dtype=torch.float32, device=self.device)
                rotation_matrix = axis_angle_to_matrix(axis_angle_tensor.unsqueeze(0)).squeeze(0)
                self.rest_positions = self.rest_positions @ rotation_matrix.T
            if "translate" in self.scene_parameters["transform"]:
                # apply translation
                translation = np.array(self.scene_parameters["transform"]["translate"])
                translation = torch.tensor(translation, dtype=torch.float32, device=self.device)
                self.rest_positions += translation
        self.positions = self.rest_positions.clone()
        self.rest_positions = self.rest_positions * self.length_conversion
        self.rest_positions = self.rest_positions.clone()
        self.original_uv = self.uv.clone()
        self.uv = self.uv.clone().requires_grad_(True)

    def initializeNetwork(self):
        network = toCuda(get_Net(params))
        logger = Logger(get_hyperparam(params), use_csv=False, use_tensorboard=False)
        print('load_date_time:', params.inference.load_date_time)
        date_time, index = logger.load_state(network, None, datetime=params.inference.load_date_time, index=params.inference.load_index, device=self.device)
        print(f"loaded: {date_time}, {index}")
        self.load_index = index
        self.cloth_net = network
        self.cloth_net.eval()

    def initializeCameraMatrix(self):
        self.crop_size = self.scene_parameters["upper_right_corner"] - self.scene_parameters["lower_left_corner"]
        self.initialize_synthetic_camera()

    def initialize_synthetic_camera(self):
        image_size = np.array([self.scene_parameters['height'], self.scene_parameters['width']])
        self.image_size = (int(image_size[0]), int(image_size[1]))
        camera_pos = torch.tensor(self.scene_parameters["camera_position"],  device=self.device)[None, :]
        up_dir = torch.tensor(self.scene_parameters["camera_up"],  device=self.device)[None, :]
        object_pos = torch.tensor(self.scene_parameters["object_position"], device=self.device)[None, :]
        R, T = look_at_view_transform(eye=camera_pos, at=object_pos, up=up_dir, device=self.device)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        self.cameras = cameras

    def initializeOptimization(self):
        self.vertex_forces = torch.zeros((1, self.simulation_frames, 3, self.h, self.w), device=self.device, dtype=torch.float32, requires_grad=True)
        self.stretching_stiffness = torch.tensor([params.inference.material.stretching], device=self.device, dtype=torch.float32, requires_grad=True)
        self.shearing_stiffness = torch.tensor([params.inference.material.shearing], device=self.device, dtype=torch.float32, requires_grad=True)
        self.bending_stiffness = torch.tensor([params.inference.material.bending], device=self.device, dtype=torch.float32, requires_grad=True)
        self.original_dataset = DatasetCloth(self.h, self.w, 1, 1,
                                             1000,
                                             iterations_per_timestep=params.inference.iterations_per_timestep,
                                             stiffness_range=params.cloth.stretching_range,
                                             shearing_range=params.cloth.shearing_range,
                                             bending_range=params.cloth.bending_range, a_ext_range=params.cloth.g)

        positions_net = self.rest_positions.transpose(0, 1).view(3, self.h, self.w).type(torch.float32)
        self.x = positions_net.clone().unsqueeze(0)
        # self.original_dataset.reset0_env(0)

        self.original_dataset.reset0_sft_env(self.x)
        self.original_dataset.set_position(positions_net)
        self.test_dataset = DatasetToSingleChannel(self.original_dataset)
        self.external_forces = self.original_dataset.g_vect.clone().detach().requires_grad_(True)
        print('initial external forces:', self.external_forces[0, 0, 0, 0].item(), self.external_forces[0, 1, 0, 0].item(),
              self.external_forces[0, 2, 0, 0].item())
        a_ext = self.external_forces + self.vertex_forces[:, self.frame_counter]
        self.original_dataset.set_optimizable(a_ext, self.stretching_stiffness, self.shearing_stiffness, self.bending_stiffness)

        self.parameters = [
            self.stretching_stiffness,
            self.shearing_stiffness,
            self.bending_stiffness,
            self.external_forces,
            self.vertex_forces
        ]

        learning_rates = [
            params.inference.sft.lr.stretching,
            params.inference.sft.lr.shearing,
            params.inference.sft.lr.bending,
            params.inference.sft.lr.external * self.length_conversion / (self.time_conversion * self.time_conversion),
            params.inference.sft.lr.vertex
        ]
        if params.inference.sft.optimize_uv:
            self.parameters = self.parameters + [self.uv]
            learning_rates = learning_rates + [params.inference.sft.lr.uv]

        self.optimizer = [None] * len(self.parameters)
        for i in range(len(self.parameters)):
            self.optimizer[i] = torch.optim.Adam([self.parameters[i]], lr=learning_rates[i])

        self.loss = torch.tensor([0.], dtype=torch.float32).to(self.device)
        self.predicted_a = torch.zeros((self.simulation_frames + 1, 3, self.h, self.w), device=self.device, dtype=torch.float32)
        self.scales = []
        self.max_scales = []

    def loadGroundTruth(self):
        self.gt_images = torch.ones((self.simulation_frames + 1,
                                     self.crop_size[0],
                                     self.crop_size[1],
                                     4),
                                    dtype=torch.float32).to(self.device)


        for i in range(self.simulation_frames + 1):
            image = torch.from_numpy(np.array(Image.open(self.scene_parameters["image_files"] + str(i).zfill(3) + ".png"), dtype = np.float32)).to(self.device)
            self.gt_images[i,:,:,:3] = image[self.scene_parameters["lower_left_corner"][0]:self.scene_parameters["upper_right_corner"][0],
                                             self.scene_parameters["lower_left_corner"][1]:self.scene_parameters["upper_right_corner"][1],
                                             :3]

            mask = torch.from_numpy(np.array(Image.open(self.scene_parameters["mask_files"] + str(i).zfill(3) + ".png"), dtype = np.float32)).to(self.device)
            if mask.dim() == 2:
                mask = mask.unsqueeze(2)
            self.gt_images[i,:,:,3] = torch.mean(mask[self.scene_parameters["lower_left_corner"][0]:self.scene_parameters["upper_right_corner"][0],
                                                      self.scene_parameters["lower_left_corner"][1]:self.scene_parameters["upper_right_corner"][1], :3], dim=2)


        self.gt_images = self.gt_images / 255.0
        self.blurred_gt_images = self.gt_images.clone()
        self.blurred_gt_images = torch.permute(self.blurred_gt_images, (0, 3, 1, 2))
        self.blurred_gt_images[:, :3, :, :] = gaussian_blur(self.blurred_gt_images[:, :3, :, :], 27, 7)
        self.blurred_gt_images = torch.permute(self.blurred_gt_images, (0, 2, 3, 1))


    def renderPyTorch3D(self, x_new):
        vertices = x_new.view(3, -1).transpose(0, 1).contiguous() / (self.length_conversion)
        return render_pytorch(
            vertices,
            self.faces,
            self.texture_image,
            self.uv,
            self.faces_uv,
            self.cameras,
            self.image_size)


    def processImages(self, image):
        image = image.squeeze(0)
        # crop image
        image = image[self.scene_parameters["lower_left_corner"][0]:self.scene_parameters["upper_right_corner"][0],
                      self.scene_parameters["lower_left_corner"][1]:self.scene_parameters["upper_right_corner"][1]]
        blurred_image = torch.permute(image, (2, 0, 1))
        blurred_image = gaussian_blur(blurred_image, 27, 7)
        blurred_image = torch.permute(blurred_image, (1, 2, 0))

        image_diff = image - self.gt_images[self.frame_counter]
        blurred_image_diff = blurred_image - self.blurred_gt_images[self.frame_counter]
        return image_diff, blurred_image_diff

    def printQuantities(self):
        t = time.perf_counter()
        if self.epoch_counter % 1 == 0:
            print(f"{self.epoch_counter:5d} | {t - self.time:.2f} s | {t - self.time_start:7.2f} s | "
                  f"Loss: {self.loss.item() * self.frames_per_epoch:.2e} {self.loss.item():.2e} | "
                  f"{self.stretching_stiffness.item():.3e} "
                  f"{self.shearing_stiffness.item():.3e} "
                  f"{self.bending_stiffness.item():.3e} "
                  f"{(self.external_forces[0, 0, 0, 0].item()): .2e} "
                  f"{(self.external_forces[0, 1, 0, 0].item()): .2e} "
                  f"{(self.external_forces[0, 2, 0, 0].item()): .2e}   "
                  f"{self.chamfer_distance.item():.2e}")
        self.time = t

    def resetState(self):
        positions_net = self.rest_positions.transpose(0, 1).view(3, self.h, self.w).type(torch.float32)
        with torch.no_grad():
        #     # self.original_dataset.reset0_env(0)
            self.original_dataset.reset_sft_env(0)
        self.original_dataset.set_position(positions_net)
        with torch.no_grad():
            self.positions[:] = self.rest_positions / self.length_conversion
            self.loss = torch.tensor([0.], dtype=torch.float32, device=self.device)
            # remove temporally and spatially constant part that could be modeled by wind
            self.vertex_forces -= torch.mean(self.vertex_forces, dim=[1, 3, 4]).unsqueeze_(1).unsqueeze_(3).unsqueeze_(4)

    def clampOptimization(self):
        with torch.no_grad():
            self.stretching_stiffness[0] = max(10, self.stretching_stiffness[0])
            self.shearing_stiffness[0] = max(1e-2, self.shearing_stiffness[0])
            self.bending_stiffness[0] = max(1e-5, self.bending_stiffness[0])
            if self.real_scene:
                self.external_forces[0][1][0][0] = -1   # original one

    def initialize(self, scene, max_epochs, evaluate):
        print("Start: Initialization")
        t_start = time.perf_counter()
        self.initializeParameters(scene, evaluate)
        self.initializeMesh()
        self.initializeNetwork()
        self.initializeCameraMatrix()
        self.initializeOptimization()

        # texture
        texture_path = self.scene_parameters["texture_file"]
        texture = np.array(Image.open(texture_path)) / 255.0        # TODO resize?
        self.texture_image = torch.tensor(texture)[None, ...].to(dtype=torch.float32).to(self.device)

        print('load ground truth...')
        self.loadGroundTruth()        ## todo uncomment
        print('done')


        t_end = time.perf_counter()
        print(f"Done:  Initialization in {t_end - t_start:.3f} s\n")
        print("Epoch |  Time  |  Total t  | Loss:   Full  per Frame |  Stretch    Shear     Bend      Wind x    Wind y    Wind z    Vertex F   uv_smooth      e3D     ")
        print("------+--------+-----------+-------------------------+------------------------------------------------------------------------------------------------")

        self.time = time.perf_counter()
        self.time_start = time.perf_counter()

    def step(self):
        # update vertex force
        a_ext = torch.ones(1, 3, self.h, self.w, device=device) * self.external_forces
        a_ext = a_ext + self.vertex_forces[:, self.frame_counter]
        self.original_dataset.set_optimizable(a_ext, self.stretching_stiffness, self.shearing_stiffness, self.bending_stiffness)
        # print('stretch:', self.stretching_stiffness[0].item())
        # print('shear:', self.shearing_stiffness[0].item())
        # print('bend:', self.bending_stiffness[0].item())

        grads, hidden_states = self.test_dataset.ask_sft()
        update_steps, new_hidden_states = self.cloth_net(grads, hidden_states)
        _ = self.test_dataset.tell_sft(update_steps, new_hidden_states)
        self.scales.append(new_hidden_states[0][2][0, 0, 0, 0].detach().cpu().numpy())

        if (self.t_iter + 1) % params.inference.iterations_per_timestep == 0:  # visualize only at a new timestep (a timestep can take several iterations to optimize)
            self.frame_counter += 1
            index = 0
            x = self.original_dataset.x[index]  # FIXME no grads?

            with torch.no_grad():
                self.positions[:] = x.view(3, -1).transpose(0, 1) / self.length_conversion

            ### DIFFRAST
            image = self.renderPyTorch3D(x.to(dtype=torch.float32))
            image_diff, blurred_image_diff = self.processImages(image)

            ### OPTIMIZATION
            self.loss += params.inference.sft.lc.rgb * torch.mean(torch.abs(image_diff[..., :3]))  # IMAGE LOSS

            if self.frame_counter == self.frames_per_epoch:
                self.chamfer_distance = torch.tensor([0.0])

                # mean over all frames
                self.loss = self.loss / self.frames_per_epoch

                self.printQuantities()
                loss_norm = self.loss.detach() + 1e-3
                self.loss = self.loss / loss_norm
                # self.loss.backward(retain_graph=True) # ok
                self.loss.backward()

                for i, p in enumerate(self.parameters):
                    print(f'param[{i}] grad =', None if p.grad is None else p.grad.mean())

                for i in range(len(self.parameters)):
                    # print grads of parameters
                    self.optimizer[i].step()
                    self.optimizer[i].zero_grad()

                self.frame_counter = 0
                self.epoch_counter += 1
                self.resetState()
                self.clampOptimization()
                # successively add frames
                if self.frames_per_epoch < self.simulation_frames and self.epoch_counter % self.new_frame_period == 0:
                    self.frames_per_epoch += 1

        self.t_iter += 1

def main():
    # set seed
    torch.manual_seed(params.data.seed)
    np.random.seed(params.data.seed)
    random.seed(params.data.seed)
    scene_list = params.inference.sft.json
    evaluate = params.inference.sft.evaluate
    for file_name in scene_list:
        scene = loadJson(file_name)
        opt = Optimization(save_render=params.inference.sft.save_render, save_mesh=params.inference.sft.save_mesh, debug=params.inference.sft.debug, device=device)
        max_epochs = params.inference.sft.n_epochs_opt + 1
        opt.initialize(scene=scene, max_epochs=max_epochs, evaluate=evaluate)
        while (opt.epoch_counter < max_epochs):
            opt.step()
            torch.cuda.empty_cache()
        print("------+--------+-----------+-------------------------+-------------------------------------------------------------------------------------")
        opt.printQuantities()


if __name__ == '__main__':
    params.wandb.log = False    # disable wandb logging
    params.training = False
    print('device:', device)
    print('params:', params)
    main()
