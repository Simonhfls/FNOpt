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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Logger import Logger
from dataset_cloth3 import DatasetCloth
from dataset_utils import DatasetToSingleChannel
from get_param2 import params, device
from metamizer import get_Net3 as get_Net
from sft import evaluation
from sft.render import opencv_projection, ComputeViewMatrix, render_pytorch, render_nvdiffrast
from sft.utils import loadJson
import numpy as np
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
from pytorch3d.io import load_obj


torch.autograd.set_detect_anomaly(False)

class Optimization():
    def __init__(self, save_render=False, save_mesh=False, debug=False, seq_desc='default_seq', device='cuda'):
        self.save_render = save_render
        self.save_mesh = save_mesh
        self.debug = debug
        self.device = device
        self.dtype = params.net.dtype
        self.seq_desc = seq_desc

    
    def initializeParameters(self, scene, evaluate):
        self.scene_parameters = scene
        self.mesh_resolution = 32 if 'mesh_resolution' not in scene else scene['mesh_resolution']
        self.real_scene = True if self.scene_parameters['scene'][0] == 'R' else False
        self.h = self.mesh_resolution # 32
        self.w = self.mesh_resolution # 32
        self.frame_counter = 0
        self.t_iter = 0
        # self.frames_per_epoch = min(3, self.scene_parameters["n_images"])
        # self.frames_per_epoch = min(49, self.scene_parameters["n_images"])
        self.frames_per_epoch = min(10, self.scene_parameters["n_images"])
        self.new_frame_period = params.inference.sft.new_frame_period
        self.epoch_counter = 0
        self.simulation_frames = self.scene_parameters["n_images"]
        self.evaluate = evaluate
        self.dpi = 200
        self.time_conversion = params.inference.sft.framerate  # 1s = 50 [NN-t]
        # note in metamizer, length_conversion is set to the resolution of the tested cloth, instead of training resolution.
        self.length_conversion = (self.mesh_resolution - 1) / self.scene_parameters["mesh_size"]  # 1m = 31 [NN-m]
        self.bc_n_x = math.ceil(self.mesh_resolution / 32)
        self.dir_dataset = scene['dataset_dir']
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
            self.dir_debug = os.path.abspath(os.path.dirname(self.scene_parameters["result_chamfer_file"]) + '/debug/' + self.seq_desc + self.unique_id)
            if not os.path.exists(self.dir_debug):
                os.makedirs(self.dir_debug)
            print('debug dir:', self.dir_debug)
        if self.save_mesh:
            self.dir_obj = os.path.dirname(os.path.dirname(self.scene_parameters['result_point_cloud_files'])) + '/obj/'
            if not os.path.exists(self.dir_obj):
                os.makedirs(self.dir_obj)
        self.renderer = params.inference.renderer.lower()



    def initializeMesh(self):
        verts, faces, aux = load_obj(str(Path(self.dir_dataset) / self.scene_parameters["mesh_file"]), load_textures=True, device=self.device)
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
        # network.nn = torch.compile(network.nn)

        logger = Logger(get_hyperparam(params), use_csv=False, use_tensorboard=False)
        print('load_date_time:', params.inference.load_date_time)
        date_time, index = logger.load_state(network, None, datetime=params.inference.load_date_time, index=params.inference.load_index, device=self.device)
        print(f"loaded: {date_time}, {index}")
        self.load_index = index
        self.cloth_net = network

        # network.nn = torch.compile(
        #     network.nn,
        #     mode="reduce-overhead")

        self.cloth_net.eval()
        for param in self.cloth_net.parameters():
            param.requires_grad = False
        # for name, param in self.cloth_net.named_parameters():
        #     if param.requires_grad:
        #         print(f"{name} requires grad")
        pass

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
                                  device=self.device,
                                  dtype=self.dtype)[None, :]
        up_dir = torch.tensor(self.scene_parameters["camera_up"],
                              device=self.device,
                              dtype=self.dtype)[None, :]
        object_pos = torch.tensor(self.scene_parameters["object_position"],
                                  device=self.device,
                                  dtype=self.dtype)[None, :]

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
        self.vertex_forces = torch.zeros((1, self.simulation_frames, 3, self.h, self.w), device=self.device, dtype=torch.float32, requires_grad=True)
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
        # self.original_dataset.reset0_env(0)

        # g = -9.81 * self.length_conversion / (self.time_conversion * self.time_conversion)  # convert to NN units
        bc_indices = None
        if "handles" in self.scene_parameters:
            bc_indices = self.scene_parameters["handles"]
        self.original_dataset.reset0_sft_env(self.x, bc_indices=bc_indices)        # todo check, no need ?
        # self.original_dataset.reset_position(positions_net)
        self.test_dataset = DatasetToSingleChannel(self.original_dataset)

        gravity = torch.tensor(self.scene_parameters["gravity"], device=self.device, dtype=torch.float32)
        self.external_forces = torch.tensor(gravity * self.length_conversion / (self.time_conversion * self.time_conversion), device=self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(2).unsqueeze(3).requires_grad_(True)
        # self.external_forces = torch.tensor([0, 0, -1]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)       # default

        # self.external_forces = self.original_dataset.g_vect.clone().detach().requires_grad_(True)
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
            self.optimizer[i] = torch.optim.Adam([self.parameters[i]], lr=learning_rates[i]) # default
            # self.optimizer[i] = torch.optim.AdamW([self.parameters[i]], lr=learning_rates[i])

        self.loss = torch.tensor([0.], dtype=torch.float32).to(self.device)
        self.predicted_a = torch.zeros((self.simulation_frames + 1, 3, self.h, self.w), device=self.device, dtype=torch.float32)
        self.scales = []

    def loadGroundTruth(self):
        self.gt_images = torch.ones((self.simulation_frames + 1,
                                     self.crop_size[0],
                                     self.crop_size[1],
                                     4),
                                    dtype=torch.float32).to(self.device)


        for i in range(self.simulation_frames + 1):
            image = torch.from_numpy(np.array(Image.open(Path(self.dir_dataset) / f"{self.scene_parameters['image_files']}{str(i).zfill(3)}.png"), dtype = np.float32)).to(self.device)

            self.gt_images[i,:,:,:3] = image[self.scene_parameters["lower_left_corner"][0]:self.scene_parameters["upper_right_corner"][0],
                                             self.scene_parameters["lower_left_corner"][1]:self.scene_parameters["upper_right_corner"][1], :3]

            mask = torch.from_numpy(np.array(Image.open(Path(self.dir_dataset) / f"{self.scene_parameters['mask_files']}{str(i).zfill(3)}.png"), dtype = np.float32)).to(self.device)
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

    def renderDiffrast(self, x_new):
        vertices = x_new.view(3, -1).transpose(0, 1).contiguous() / (self.length_conversion)
        uvs = torch.concat([self.uv], axis=-1).unsqueeze(0)

        return render_nvdiffrast(self.context,
                                 vertices,
                                 self.faces,
                                 uvs,
                                 self.texture_image,
                                 self.cameras,
                                 self.image_size, real_scene=self.real_scene)

    def processImages(self, image):
        if self.renderer == "nvdiffrast":
            image = torch.flip(image, dims=[2])

        if self.epoch_counter % 1 == 0 and self.save_render:
            img = image.squeeze(0).cpu().detach().numpy()
            img = Image.fromarray((img * 255).astype(np.uint8))
            save_path = os.path.join(self.dir_rgb, str(self.frame_counter).zfill(3) + ".png")
            img.save(save_path)

        image = image.squeeze(0)
        # crop image
        image = image[self.scene_parameters["lower_left_corner"][0]:self.scene_parameters["upper_right_corner"][0],
                      self.scene_parameters["lower_left_corner"][1]:self.scene_parameters["upper_right_corner"][1]]
        blurred_image = torch.permute(image, (2, 0, 1))
        blurred_image = gaussian_blur(blurred_image, 27, 7)
        blurred_image = torch.permute(blurred_image, (1, 2, 0))

        image_diff = image - self.gt_images[self.frame_counter]
        blurred_image_diff = blurred_image - self.blurred_gt_images[self.frame_counter]

        if self.debug and (self.epoch_counter % params.inference.sft.debug_n_save == 0 or self.epoch_counter == params.inference.sft.n_epochs_opt - 1):
            # save gt_image
            # gt_img = self.gt_images[self.frame_counter].detach().cpu().numpy()
            # gt_img = Image.fromarray((gt_img * 255).astype(np.uint8))
            # gt_img.save(os.path.join(self.dir_debug, f'01_gt_image_{str(self.frame_counter).zfill(3)}.png'))

            img = image.squeeze(0).detach().cpu().numpy()
            img = Image.fromarray((img * 255).astype(np.uint8))
            img.save(os.path.join(self.dir_debug, f'rgb_{str(self.frame_counter).zfill(3)}.png'))
            # img.save(os.path.join(self.dir_debug, f'{str(self.frame_counter).zfill(3)}.png'))

            sil = blurred_image[..., 3].detach().cpu().numpy()
            sil = Image.fromarray((sil * 255).astype(np.uint8))
            sil.save(os.path.join(self.dir_debug, f'blurred_mask_{str(self.frame_counter).zfill(3)}.png'))

            img_diff = torch.abs(image_diff[..., :3]).detach().cpu().numpy()
            img_diff = Image.fromarray((img_diff * 255).astype(np.uint8))
            img_diff.save(os.path.join(self.dir_debug, f'03_image_diff_{str(self.frame_counter).zfill(3)}.png'))

            silhouette_diff = torch.abs(blurred_image_diff[..., 3]).detach().cpu().numpy()
            silhouette_diff = Image.fromarray((silhouette_diff * 255).astype(np.uint8))
            silhouette_diff.save(
                os.path.join(self.dir_debug, f'04_silhouette_diff_{str(self.frame_counter).zfill(3)}.png'))

            # save blurred_image
            img = blurred_image.squeeze(0).detach().cpu().numpy()
            img = Image.fromarray((img * 255).astype(np.uint8))
            img.save(os.path.join(self.dir_debug, f'05_blurred_image_{str(self.frame_counter).zfill(3)}.png'))

            img = self.blurred_gt_images[self.frame_counter].squeeze(0).detach().cpu().numpy()
            img = Image.fromarray((img * 255).astype(np.uint8))
            img.save(os.path.join(self.dir_debug, f'06_blurred_gt_image_{str(self.frame_counter).zfill(3)}.png'))

        return image_diff, blurred_image_diff

    def printQuantities(self):
        t = time.perf_counter()
        if self.epoch_counter % 1 == 0:
            print(f"{self.epoch_counter:5d} | {t - self.time:6.2f} s | {t - self.time_start:7.2f} s | "
                  f"Loss: {self.loss.item() * self.frames_per_epoch:.2e} {self.loss.item():.2e} | "
                  f"{self.stretching_stiffness.item():.3e} "
                  f"{self.shearing_stiffness.item():.3e} "
                  f"{self.bending_stiffness.item():.3e} "
                  f"{(self.external_forces[0, 0, 0, 0].item() / self.length_conversion * self.time_conversion * self.time_conversion): .2e} "
                  f"{(self.external_forces[0, 1, 0, 0].item() / self.length_conversion * self.time_conversion * self.time_conversion): .2e} "
                  f"{(self.external_forces[0, 2, 0, 0].item() / self.length_conversion * self.time_conversion * self.time_conversion): .2e}   "
                  f"{self.vertex_shift_loss.item():.2e}    "
                  f"{self.uv_smooth_loss.item():.2e}    "
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
            # remove temporally and spatially constant part that could be modeled by wind
            # self.vertex_forces -= torch.mean(self.vertex_forces, dim=[1, 3, 4]).unsqueeze_(1).unsqueeze_(3).unsqueeze_(4)
            # Note Modification3: I compute average vertex force on only already predicted frames instead of all frames
            mean_vf = self.vertex_forces[:, :self.frames_per_epoch].mean(dim=[1, 3, 4], keepdim=True)
            self.vertex_forces -= mean_vf
            # Note Modification4: I added this to add the subtracted forces to the external forces
            #  -> Improves the results a bit
            self.external_forces += mean_vf[0]        # TODO add this
            pass

    def clampOptimization(self):
        with torch.no_grad():
            self.stretching_stiffness[0] = max(10, self.stretching_stiffness[0])
            self.shearing_stiffness[0] = max(1e-2, self.shearing_stiffness[0])
            self.bending_stiffness[0] = max(1e-5, self.bending_stiffness[0])
            if self.real_scene:
                self.external_forces[0][1][0][0] = -9.81 * self.length_conversion / (self.time_conversion * self.time_conversion)   # original one


    def initialize(self, scene, max_epochs, evaluate):
        print("Start: Initialization")
        t_start = time.perf_counter()
        self.initializeParameters(scene, evaluate)
        self.initializeMesh()
        self.initializeNetwork()
        self.initializeCameraMatrix()
        self.initializeOptimization()
        if self.renderer == "nvdiffrast":
            self.context = dr.RasterizeCudaContext()

        # texture
        texture_path = str(Path(self.dir_dataset) / self.scene_parameters["texture_file"])
        texture = np.array(Image.open(texture_path)) / 255.0        # TODO resize?
        self.texture_image = torch.tensor(texture)[None, ...].to(dtype=torch.float32).to(self.device)
        if self.renderer == "nvdiffrast":
            self.texture_image = torch.flip(self.texture_image, dims=[1])

        print('load ground truth...', end='')
        self.loadGroundTruth()
        print('done')

        if evaluate:
            self.chamfer_distance = torch.tensor([0.0])
            self.chamfer_distances_epochs = torch.zeros((max_epochs), device=self.device)
            if self.real_scene:
                ground_truth_point_clouds, point_clouds_lengths = evaluation.loadGroundTruth(self.scene_parameters, device=self.device)
                our_point_clouds = torch.zeros_like(ground_truth_point_clouds)
                self.point_clouds = {"ground_truth": ground_truth_point_clouds, "ours": our_point_clouds,
                                     "lengths": point_clouds_lengths}
            else:
                self.gt_x = torch.zeros((self.simulation_frames + 1, self.h * self.w, 3), dtype=torch.float32).to(
                    self.device)
                for i in range(self.simulation_frames + 1):
                    self.gt_x[i], _, _ = load_obj(str(Path(self.dir_dataset) / self.scene_parameters["ground_truth_dir"] / f"{i:04d}_00.obj"))

                our_v = torch.zeros_like(self.gt_x)
                self.point_clouds = {"ground_truth": self.gt_x, "ours": our_v}

        t_end = time.perf_counter()
        print(f"Done:  Initialization in {t_end - t_start:.3f} s\n")
        if self.real_scene:
            print("Epoch |  Time   |  Total t  | Loss:   Full  per Frame |  Stretch    Shear     Bend      Wind x    Wind y    Wind z    Vertex F   uv_smooth  Chamfer dist")
        else:
            print("Epoch |  Time   |  Total t  | Loss:   Full  per Frame |  Stretch    Shear     Bend      Wind x    Wind y    Wind z    Vertex F   uv_smooth      e3D     ")
        print("------+--------+-----------+-------------------------+------------------------------------------------------------------------------------------------")

        self.time = time.perf_counter()
        self.time_start = time.perf_counter()

    # @torch.jit.script
    def step(self):
        # update vertex force
        a_ext = self.external_forces + self.vertex_forces[:, self.frame_counter]
        self.original_dataset.set_optimizable(a_ext, self.stretching_stiffness, self.shearing_stiffness, self.bending_stiffness)

        last_iter = ((self.t_iter + 1) % params.inference.sft.iterations_per_timestep == 0)
        grads, hidden_states = self.test_dataset.ask_sft(last_iter=last_iter)
        update_steps, new_hidden_states = self.cloth_net(grads, hidden_states)
        _ = self.test_dataset.tell_sft(update_steps, new_hidden_states)
        if params.inference.visualize_scaling:
            self.scales.append(new_hidden_states[0][2][0, 0, 0, 0].detach().cpu().numpy())
        if last_iter:  # visualize only at a new timestep (a timestep can take several iterations to optimize)
            self.frame_counter += 1
            index = 0
            x = self.original_dataset.x[index]  # FIXME no grads?
            with torch.no_grad():
                self.positions[:] = x.view(3, -1).transpose(0, 1) / self.length_conversion
                if self.evaluate:
                    save_dir = self.scene_parameters["result_point_cloud_files"]
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    if self.real_scene:
                        self.point_clouds["ours"][self.frame_counter,:self.point_clouds["lengths"][self.frame_counter]] = evaluation.sampleMesh(self.point_clouds["lengths"][self.frame_counter].item(), self.positions,self.faces.type(torch.long),device=self.device)
                        if self.epoch_counter == params.inference.sft.n_epochs_opt - 1:
                            torch.save(self.point_clouds["ours"][self.frame_counter,:self.point_clouds["lengths"][self.frame_counter]].clone(), save_dir + str(self.frame_counter).zfill(3) + ".pt")
                    else:
                        self.point_clouds["ours"][self.frame_counter] = self.positions
                        if self.epoch_counter == params.inference.sft.n_epochs_opt - 1:
                            torch.save(self.point_clouds["ours"][self.frame_counter].clone(), save_dir + str(self.frame_counter).zfill(3) + ".pt")
            ### DIFFRAST
            if self.renderer == "pytorch3d":
                image = self.renderPyTorch3D(x.to(dtype=torch.float32))
            elif self.renderer == "nvdiffrast":
                image = self.renderDiffrast(x.to(dtype=torch.float32))
            image_diff, blurred_image_diff = self.processImages(image)

            ### OPTIMIZATION
            self.loss += params.inference.sft.lc.rgb * torch.mean(torch.abs(image_diff[..., :3]))  # IMAGE LOSS
            self.loss += params.inference.sft.lc.sil * torch.mean(torch.abs(blurred_image_diff[..., 3]))  # SILHOUETTE LOSS

            # uv consistency loss
            uv = self.uv.view(self.h, self.w, 2)
            uv_diff_h = uv[:, 1:, :] - uv[:, :-1, :]
            uv_diff_v = uv[1:, :, :] - uv[:-1, :, :]
            self.uv_smooth_loss = torch.mean(uv_diff_h ** 2) + torch.mean(uv_diff_v ** 2)
            self.loss += params.inference.sft.lc.uv_smooth * self.uv_smooth_loss

            # print('frame counter:', self.frame_counter, 'frames_per_epoch:', self.frames_per_epoch)
            if self.frame_counter == self.frames_per_epoch:

                self.chamfer_distance = torch.tensor([0.0])
                if self.evaluate:
                    with torch.no_grad():
                        if self.real_scene:
                            self.chamfer_distance = evaluation.computeChamferDistance(self.point_clouds["ground_truth"],
                                                                                  self.point_clouds["ours"], 0,
                                                                                  self.frame_counter + 1,
                                                                                  self.point_clouds["lengths"])
                        else:
                            self.chamfer_distance = evaluation.computeMeshDistance(self.point_clouds["ground_truth"], self.point_clouds["ours"], 0, self.frame_counter+1)
                        self.chamfer_distances_epochs[self.epoch_counter] = self.chamfer_distance

                # mean over all frames
                self.loss = self.loss / self.frames_per_epoch

                # VERTEX SHIFT REGULARIZATION
                self.vertex_shift_loss = torch.zeros(1).to(self.device)

                exponent = 2
                self.vertex_shift_loss += torch.mean(torch.linalg.norm(self.vertex_forces[:, :self.frames_per_epoch], dim=2) ** exponent)
                self.vertex_shift_loss += torch.mean(torch.linalg.norm(self.vertex_forces[:, 1:self.frames_per_epoch] - self.vertex_forces[:, :self.frames_per_epoch - 1], dim=2) ** exponent)
                self.vertex_shift_loss += 0.1 * torch.mean(torch.linalg.norm(self.vertex_forces[:, :self.frames_per_epoch, :, 1:] - self.vertex_forces[:, :self.frames_per_epoch, :, :-1], dim=2) ** exponent)
                self.vertex_shift_loss += 0.1 * torch.mean(torch.linalg.norm(self.vertex_forces[:, :self.frames_per_epoch, :, :, 1:] - self.vertex_forces[:, :self.frames_per_epoch, :, :, :-1], dim=2) ** exponent)
                self.loss += params.inference.sft.lc.shift * self.vertex_shift_loss

                self.printQuantities()

                # NOTE modification2: I deleted loss normalization, seems improving the results.
                # loss_norm = self.loss.detach() + 1e-3
                # self.loss = self.loss / loss_norm
                self.loss.backward()

                for i in range(len(self.parameters)):
                    # print grads of parameters
                    self.optimizer[i].step()
                    self.optimizer[i].zero_grad()

                # for i, p in enumerate(self.parameters):
                #     print(f"After:  param[{i}] = {p.item() if p.numel() == 1 else p.flatten()[0].item()}")
                torch.cuda.empty_cache()
                # allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                # reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # MB
                # print(f" Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")

                self.frame_counter = 0
                self.epoch_counter += 1

                self.resetState()
                self.clampOptimization()

                # successively add frames
                if self.frames_per_epoch < self.simulation_frames and self.epoch_counter % self.new_frame_period == 0:
                    print('here add new frame')
                    self.frames_per_epoch += 1
                    if self.frames_per_epoch == self.simulation_frames:
                        print("Reached max frames")

            if params.inference.visualize_scaling:  # visualize, how scaling changes during update steps
                scales = np.stack(self.scales)
                plt.figure(2)
                plt.clf()
                stride = 1  # len(scales)//200+1
                plt.semilogy(scales[::stride])
                plt.xlabel("iteration")
                plt.ylabel("scale")
                plt.legend(["scales"])
                plt.title(
                    f"Scale of Metamizer, {params.inference.iterations_per_timestep} iterations per timestep, {self.h} x {self.w}")

                plt.draw()
                plt.pause(0.01)
                # save
                # if self.save_render and (self.epoch_counter % params.inference.sft.debug_n_save == 0 or self.epoch_counter == params.inference.sft.n_epochs_opt - 1):
                plt.savefig(os.path.join(self.dir_debug, f'scale.png'))
                if self.epoch_counter % 5 == 0:
                    plt.savefig(os.path.join(self.dir_debug, f'scale_{self.epoch_counter:04d}.png'))

        self.t_iter += 1


def main():
    # set seed
    torch.manual_seed(params.data.seed)
    np.random.seed(params.data.seed)
    random.seed(params.data.seed)
    scene_list = params.inference.sft.json
    evaluate = params.inference.sft.evaluate
    print('\n\n\n**************************')
    print("Seed: ", params.data.seed)
    print('device:', device)
    print('model:', params.net.name)
    print('renderer:', params.inference.renderer.lower())
    print('N_iter_per_step:', params.inference.sft.iterations_per_timestep)
    print('dtype:', params.net.dtype)
    print('save render:', params.inference.sft.save_render)
    print('debug:', params.inference.sft.debug)
    print('**************************\n\n\n')


    for file_name in scene_list:
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

        # 降低 matplotlib 日志级别
        # seq_desc = os.path.basename(file_name).split('.')[0]
        seq_desc = f"{params.net.name}_{scene['mesh_resolution']}" if 'mesh_resolution' in scene else f"{params.net.name}_32"
        time1 = time.perf_counter()
        opt = Optimization(save_render=params.inference.sft.save_render, save_mesh=params.inference.sft.save_mesh, debug=params.inference.sft.debug, seq_desc=seq_desc, device=device)
        max_epochs = params.inference.sft.n_epochs_opt + 1
        opt.initialize(scene=scene, max_epochs=max_epochs, evaluate=evaluate)
        # with profile(
        #         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        #         record_shapes=True,
        #         with_stack=True
        # ) as prof:
        #     with record_function("TrainStep"):
        #         while (opt.epoch_counter < max_epochs):
        #             opt.step()
        #             if opt.epoch_counter == 1:
        #                 break
        # print(prof.key_averages(group_by_stack_n=5).table(sort_by="cuda_time_total", row_limit=20))
        # print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_time_total", row_limit=20))
        # prof.export_chrome_trace("trace.json")

        while (opt.epoch_counter < max_epochs):
            opt.step()

        print("------+---------+-----------+-------------------------+-------------------------------------------------------------------------------------")
        opt.printQuantities()
        logging.info(
            f"Model:         {params.net.name}{params.inference.postfix}\n"
            f"Iters:         {params.inference.sft.iterations_per_timestep:<3}     "
            f"n_epochs:      {params.inference.sft.n_epochs_opt:<4}     "
            f"new_f_period:  {params.inference.sft.new_frame_period:<3}     "
            f"optimize_uv:   {params.inference.sft.optimize_uv:<3}\n"
            f"lr_Y:          {params.inference.sft.lr.stretching:<8.1e} "
            f"lr_S:          {params.inference.sft.lr.shearing:<8.1e} "
            f"lr_B:          {params.inference.sft.lr.bending:<8.1e}\n"
            f"lr_external:   {params.inference.sft.lr.external:<8.1e} "
            f"lr_vertex:     {params.inference.sft.lr.vertex:<8.1e} "
            f"lr_uv:         {params.inference.sft.lr.uv:<8.1e}\n"
            f"lc_rgb:        {params.inference.sft.lc.rgb:<8.1e} "
            f"lc_sil:        {params.inference.sft.lc.sil:<8.1e} "
            f"lc_shift:      {params.inference.sft.lc.shift:<8.1e}\n"
            f"Chamfer (e3d): {opt.chamfer_distance.item():.2e}"
        )

        if evaluate:
            torch.save(opt.chamfer_distances_epochs, scene["result_chamfer_file"])

        time2 = time.perf_counter()
        print(f"Done in {(time2 - time1): .2f} s")

        # TODO create a function for generating video
        if params.inference.sft.save_render:
            print(f"Generating render video for resolution {opt.mesh_resolution}")
            render_dir = opt.dir_rgb
            output_file = f"RGB_{params.net}{params.inference.postfix}_{opt.mesh_resolution}_ep{opt.load_index}.mp4"

            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-framerate", f"{params.inference.sft.framerate}",
                "-pattern_type", "glob",
                "-i", f"{render_dir}/*.png",
                "-frames:v", str(opt.simulation_frames),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                os.path.join(render_dir, output_file)
            ]

            # execute ffmpeg to render images
            try:
                start_time = time.perf_counter()
                subprocess.run(ffmpeg_cmd, check=True)
                end_time = time.perf_counter()
                print(f"Render video generation completed in {(end_time - start_time):.2f} seconds")
            except subprocess.CalledProcessError as e:
                print(f"Error generating video: {e}")

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

        if params.inference.sft.debug:
            acc_render_dir = opt.dir_debug
            output_file = f"V_RGB_GT_{opt.mesh_resolution}.mp4"

            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-framerate", f"{params.inference.sft.framerate}",
                "-pattern_type", "glob",
                "-i", f"{acc_render_dir}/01_*.png",
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

            output_file = f"V_RGB_PRED_{params.net}{params.inference.postfix}_{opt.mesh_resolution}_ep{opt.load_index}_{opt.renderer}.mp4"
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-framerate", f"{params.inference.sft.framerate}",
                "-pattern_type", "glob",
                "-i", f"{acc_render_dir}/02_*.png",
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

            output_file = f"V_RGB_DIFF_{params.net}{params.inference.postfix}_{opt.mesh_resolution}_ep{opt.load_index}.mp4"
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-framerate", f"{params.inference.sft.framerate}",
                "-pattern_type", "glob",
                "-i", f"{acc_render_dir}/03_*.png",
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

            output_file = f"V_SIL_DIFF_{params.net}{params.inference.postfix}_{opt.mesh_resolution}_ep{opt.load_index}.mp4"
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-framerate", f"{params.inference.sft.framerate}",
                "-pattern_type", "glob",
                "-i", f"{acc_render_dir}/04_*.png",
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
