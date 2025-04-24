import math
import os
import sys

from dataset_utils import DatasetToSingleChannel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from matplotlib.colors import LightSource
from Logger import Logger
from dataset_cloth2 import DatasetCloth
from get_param2 import get_params, toCuda, get_hyperparam, params
from metamizer import get_Net2 as get_Net
from sft import evaluation
from sft.render import opencv_projection, ComputeViewMatrix, render_pytorch, render_nvdiffrast
from sft.utils import loadJson, grid_to_trimesh_faces

import subprocess
import numpy as np
from pytorch3d.transforms import axis_angle_to_matrix
import time
from PIL import Image
import matplotlib.pyplot as plt
import torch
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    MeshRasterizer,
    RasterizationSettings,
    HardPhongShader,
    PointLights,
    BlendParams,
    TexturesUV, PerspectiveCameras,
)
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj, save_obj


class MeshRendererWithDepth(torch.nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf


def render(vertices, faces, texture, uvs, faces_uvs, cameras, image_size):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tex = TexturesUV(verts_uvs=uvs.unsqueeze(0), faces_uvs=faces_uvs.unsqueeze(0), maps=texture)

    meshes = Meshes(verts=[vertices], faces=[faces], textures=tex)

    sigma = 1e-4
    gamma = 1e-4

    background_color = (1.0, 1.0, 1.0)
    blend_params = BlendParams(sigma, gamma, background_color=background_color)

    renderer = MeshRendererWithDepth(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=RasterizationSettings(
                image_size=image_size,
                blur_radius=0.0,
                faces_per_pixel=1,
            )
        ),
        shader=HardPhongShader(
            device=device,
            lights=PointLights(device=device, location=[[0.0, 0.0, -3.0]]),
            cameras=cameras,
            blend_params=blend_params
        )
    )

    images = []
    depths = []
    for i in range(0, len(meshes)):
        image, depth = renderer(meshes[i])
        images.append(image)
        depths.append(depth)
    images = torch.cat(images, dim=0)

    return images


class Rollout():
    def __init__(self, simulation_frames=500, save_render=False, device='cuda'):
        self.simulation_frames = simulation_frames
        self.save_render = save_render
        self.device = device

    def initializeParameters(self, scene, evaluate):
        self.scene_parameters = scene
        self.mesh_resolution = scene['mesh_resolution']
        self.real_scene = True if self.scene_parameters['scene'][0] == 'R' else False

        self.h = self.mesh_resolution  # 32
        self.w = self.mesh_resolution  # 32
        self.frame_counter = 0
        self.t_iter = 0
        self.epoch_counter = 0
        self.evaluate = evaluate
        self.dpi = 200

        self.time_conversion = 1 / 0.02  # 1s = 50 [NN-t]
        # self.length_conversion = (self.h - 1) / self.scene_parameters["mesh_size"]  # 1m = 31 [NN-m]

        # note in metamizer, length_conversion is set to the resolution of the tested cloth, instead of training resolution.
        self.length_conversion = (self.mesh_resolution - 1) / self.scene_parameters["mesh_size"]  # 1m = 31 [NN-m]
        print('length_conversion:', self.length_conversion)

        self.bc_n_x = math.ceil(self.mesh_resolution / 32)
        self.dir_rgb = os.path.abspath(os.path.dirname(self.scene_parameters["result_chamfer_file"]) + '/rendered_fno_test/')
        if self.save_render and not os.path.exists(self.dir_rgb):
            os.makedirs(self.dir_rgb)
        self.dt = params.cloth.dt

        self.renderer = params.inference.renderer.lower()
        print('renderer:', self.renderer)


    def initializeMesh(self):
        verts, faces, aux = load_obj(self.scene_parameters["mesh_file"], load_textures=True, device=self.device)

        self.rest_positions = verts.to(dtype=torch.float32)
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
                axis_angle_tensor = torch.tensor(axis_angle, dtype=torch.float32, device=self.device)
                rotation_matrix = axis_angle_to_matrix(axis_angle_tensor.unsqueeze(0)).squeeze(0)
                self.rest_positions = self.rest_positions @ rotation_matrix.T
            if "translate" in self.scene_parameters["transform"]:
                # apply translation
                translation = np.array(self.scene_parameters["transform"]["translate"])
                translation = torch.tensor(translation, dtype=torch.float32, device=self.device)
                self.rest_positions += translation

        self.rest_positions = self.rest_positions * self.length_conversion
        self.rest_positions = self.rest_positions.clone()
        self.original_uv = self.uv.clone()
        self.uv = self.uv.clone()


    def initializeNetwork(self):
        network = toCuda(get_Net(params))

        logger = Logger(get_hyperparam(params), use_csv=False, use_tensorboard=False)

        print('load_date_time:', params.inference.load_date_time)
        date_time, index = logger.load_state(network, None, datetime=params.inference.load_date_time, index=params.inference.load_index, device=self.device)
        print(f"loaded: {date_time}, {index}")
        self.load_index = index
        self.cloth_net = network
        self.cloth_net.eval()

        self.positions_net = self.rest_positions.transpose(0, 1).view(3, self.h, self.w).type(torch.float32)
        velocities_net = torch.zeros(3, self.h, self.w, device=self.device)
        self.x_v = torch.cat([self.positions_net, velocities_net]).unsqueeze(0)
        self.original_dataset = DatasetCloth(self.h, self.w, 1, 1,
                                         params.inference.rollout.n_frames,
                                         iterations_per_timestep=params.inference.iterations_per_timestep,
                                         stiffness_range=params.cloth.stretching_range,
                                         shearing_range=params.cloth.shearing_range,
                                         bending_range=params.cloth.bending_range, a_ext_range=params.cloth.g)


        self.test_dataset = DatasetToSingleChannel(self.original_dataset)

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

        camera_pos = torch.tensor(self.scene_parameters["camera_position"], device=self.device)[
                     None, :]
        up_dir = torch.tensor(self.scene_parameters["camera_up"],  device=self.device)[None, :]
        object_pos = torch.tensor(self.scene_parameters["object_position"], device=self.device)[
                     None, :]

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
        self.external_forces = torch.tensor(
            gravity * self.length_conversion / (self.time_conversion * self.time_conversion), device=self.device,
            dtype=torch.float32).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        print('self.external_forces:', self.external_forces)
        self.vertex_forces = torch.zeros((1, self.simulation_frames, 3, self.h, self.w), device=self.device, dtype=torch.float32)
        self.predicted_a = torch.zeros((self.simulation_frames + 1, 3, self.h, self.w), device=self.device, dtype=torch.float32)
        self.scales = []
        self.max_scales = []
        self.gradients = []

        self.stretching_stiffness = torch.tensor([params.inference.material.stretching], device=self.device, dtype=torch.float32)
        self.shearing_stiffness = torch.tensor([params.inference.material.shearing], device=self.device, dtype=torch.float32)
        self.bending_stiffness = torch.tensor([params.inference.material.bending], device=self.device, dtype=torch.float32)


        self.original_dataset.reset0_sft_env(self.positions_net)
        # TODO delete this
        # self.external_forces = torch.zeros_like(self.external_forces)
        a_ext = self.external_forces

        self.original_dataset.set_optimizable(a_ext, self.stretching_stiffness, self.shearing_stiffness, self.bending_stiffness)
        self.original_dataset.set_materials(self.stretching_stiffness, self.shearing_stiffness, self.bending_stiffness)


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
        diffrast_attributes = torch.concat([self.uv], axis=-1).unsqueeze(0)
        return render_nvdiffrast(self.context,
                                 vertices,
                                 self.faces,
                                 diffrast_attributes,
                                 self.texture_image,
                                 self.cameras,
                                 self.image_size,
                                 real_scene=self.real_scene)

    def printQuantities(self):
        t = time.perf_counter()
        if self.epoch_counter % 1 == 0:
            print(f"{self.epoch_counter:5d} | {t - self.time:.2f} s | {t - self.time_start:7.2f} s | "
                  f"{self.stretching_stiffness.item():.3e} "
                  f"{self.shearing_stiffness.item():.3e} "
                  f"{self.bending_stiffness.item():.3e} "
                  f"{(self.external_forces[0, 0, 0, 0].item() / self.length_conversion * self.time_conversion * self.time_conversion): .2e} "
                  f"{(self.external_forces[0, 1, 0, 0].item() / self.length_conversion * self.time_conversion * self.time_conversion): .2e} "
                  f"{(self.external_forces[0, 2, 0, 0].item() / self.length_conversion * self.time_conversion * self.time_conversion): .2e} ")
        self.time = t

    def initialize(self, scene, evaluate):
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
        texture_path = self.scene_parameters["texture_file"]
        texture = np.array(Image.open(texture_path)) / 255.0        # TODO resize?
        self.texture_image = torch.tensor(texture)[None, ...].to(dtype=torch.float32).to(self.device)
        if self.renderer == "nvdiffrast":
            self.texture_image = torch.flip(self.texture_image, dims=[1])

        if evaluate:
            if self.real_scene:
                ground_truth_point_clouds, point_clouds_lengths = evaluation.loadGroundTruth(self.scene_parameters, device=self.device)
                our_point_clouds = torch.zeros_like(ground_truth_point_clouds)
                self.point_clouds = {"ground_truth": ground_truth_point_clouds, "ours": our_point_clouds,
                                     "lengths": point_clouds_lengths}
            else:
                self.gt_x = torch.zeros((self.simulation_frames + 1, self.h * self.w, 3), dtype=torch.float32).to(
                    self.device)
                for i in range(self.simulation_frames + 1):
                    self.gt_x[i], _, _ = load_obj(self.scene_parameters["ground_truth_dir"] + str(i).zfill(4) + "_00.obj")
            our_v = torch.zeros_like(self.gt_x)
            self.point_clouds = {"ground_truth": self.gt_x, "ours": our_v}

        t_end = time.perf_counter()
        print(f"Done:  Initialization in {t_end - t_start:.3f} s\n")
        print(
            "Epoch |  Time  |  Total t  | Loss:   Full  per Frame |  Stretch    Shear     Bend      Wind x    Wind y    Wind z    Vertex F      e3D     ")
        print(
            "------+--------+-----------+-------------------------+-------------------------------------------------------------------------------------")

        self.time = time.perf_counter()
        self.time_start = time.perf_counter()

    def step(self):
        self.original_dataset.set_optimizable(self.external_forces, self.stretching_stiffness, self.shearing_stiffness, self.bending_stiffness)
        grads, hidden_states = self.test_dataset.ask()
        update_steps, new_hidden_states = self.cloth_net(grads, hidden_states)
        _ = self.test_dataset.tell(update_steps, new_hidden_states)

        self.scales.append(new_hidden_states[0][2][0, 0, 0, 0].detach().cpu().numpy())
        self.gradients.append(torch.norm(grads, p=2).detach().cpu().numpy())
        if (self.t_iter + 1) % params.inference.iterations_per_timestep == 0:  # visualize only at a new timestep (a timestep can take several iterations to optimize)
            self.frame_counter += 1

            if True:  # visualize 3D cloth
                index = 0
                x = self.original_dataset.x[index]
                x_np = x.cpu().numpy()
                a = self.original_dataset.a[index]
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

                if self.save_render:
                    path = f"plots/{get_hyperparam(params).replace(' ', '_').replace(';', '_')}/cloth/{params.inference.load_date_time}/stiff_{params.inference.material.stretching} shear_{params.inference.material.shearing} bend_{params.inference.material.bending}"
                    os.makedirs(path, exist_ok=True)
                    assert os.path.exists(path), f"Failed to create path: {path}"

                    plt.savefig(f"{path}/frame_{str(self.frame_counter).zfill(4)}.png", dpi=self.dpi)

                plt.draw()
                plt.pause(0.01)

                if params.inference.save_obj:
                    # save mesh
                    mesh_path = f"{path}/frame_{str(self.frame_counter).zfill(4)}.obj"
                    print('mesh_path:', mesh_path)
                    save_obj(mesh_path, torch.from_numpy(x).permute(1, 2, 0).reshape(-1, 3), torch.from_numpy(grid_to_trimesh_faces(num_rows=params.inference.height, num_cols=params.inference.width)))

            if params.inference.visualize_scaling:  # visualize, how scaling changes during update steps
                plt.figure(2)
                plt.clf()
                stride = 1  # len(scales)//200+1
                plt.semilogy(self.scales[::stride])
                plt.xlabel("iteration")
                plt.ylabel("scale")
                plt.legend(["scales"])
                plt.draw()
                plt.pause(0.01)

            if params.inference.visualize_grads:  # visualize, how norm of loss gradients changes during update steps
                plt.figure(3, figsize=(1600 / self.dpi, 800 / self.dpi), dpi=self.dpi)
                plt.clf()
                stride = 1  # len(scales)//200+1
                plt.semilogy(self.gradients[::stride])
                plt.xlabel("iteration")
                plt.ylabel("gradient norm")
                plt.title(
                    f"Gradient Norm reduction of Metamizer, {params.inference.iterations_per_timestep} iterations per timestep, {params.inference.height} x {params.inference.width}")
                plt.draw()
                plt.pause(0.01)

            # Render image
            if self.renderer == "pytorch3d":
                # x to float 32
                print('x:', x)
                image = self.renderPyTorch3D(x.to(dtype=torch.float32))
            elif self.renderer == "nvdiffrast":
                image = self.renderDiffrast(x.to(dtype=torch.float32))
                image = torch.flip(image, dims=[2])

            # save rendered image
            image = image.squeeze(0).cpu().detach().numpy()
            image = Image.fromarray((image * 255).astype(np.uint8))
            save_path = os.path.join(self.dir_rgb, str(self.frame_counter).zfill(4) + ".png")
            print('save path:', save_path)
            image.save(save_path)

        self.t_iter += 1


    def save_acc_visu(self):
        a_all = self.predicted_a.cpu().detach().numpy()
        a_all = np.clip(a_all, np.percentile(a_all, 5), np.percentile(a_all, 95))
        vmin_global = np.min(a_all)
        vmax_global = np.max(a_all)
        self.dir_acc = os.path.abspath(os.path.dirname(self.scene_parameters["result_chamfer_file"]) + '/rendered_acc/')
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
    scene_list = params.inference.rollout.json
    simulation_frames = params.inference.rollout.n_frames
    device = torch.device("cuda" if params.inference.cuda else "cpu")
    for file_name in scene_list:
        print('file_name:', file_name)
        scene = loadJson(file_name)
        time1 = time.perf_counter()
        opt = Rollout(simulation_frames=simulation_frames, save_render=params.inference.rollout.save_render, device=device)
        opt.initialize(scene=scene, evaluate=False)
        while (opt.t_iter < simulation_frames * params.inference.iterations_per_timestep):
            opt.step()
        print(
            "------+--------+-----------+-------------------------+-------------------------------------------------------------------------------------")
        opt.printQuantities()
        time2 = time.perf_counter()
        print(f"Done in {(time2 - time1): .2f} s")

        if params.inference.rollout.save_render:
            print(f"Generating render video for resolution {opt.mesh_resolution}")
            render_dir = opt.dir_rgb
            output_file = f"V_RGB_{params.net.name}{params.inference.postfix}_RES{opt.mesh_resolution}_Y{params.inference.material.stretching}_S{params.inference.material.shearing}_B{params.inference.material.bending}_EP{opt.load_index}_FPS{params.inference.framerate}_{params.inference.renderer}.mp4"

            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-framerate", f"{params.inference.framerate}",
                "-pattern_type", "glob",
                "-i", f"{render_dir}/*.png",
                "-frames:v", str(params.inference.rollout.n_frames),
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
            # delete all png files in render directory
            for file in os.listdir(render_dir):
                if file.endswith(".png"):
                    os.remove(os.path.join(render_dir, file))

        if params.inference.rollout.save_acc_heatmap:
            opt.save_acc_visu()
            print(f"Generating accuracy heatmap video for resolution {opt.mesh_resolution}")
            acc_render_dir = opt.dir_acc
            output_file = f"V_ACC_{params.net.name}{params.inference.postfix}_RES{opt.mesh_resolution}_Y{params.inference.material.stretching}_S{params.inference.material.shearing}_B{params.inference.material.bending}_EP{opt.load_index}_FPS{params.inference.framerate}.mp4"

            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-framerate", f"{params.inference.framerate}",
                "-pattern_type", "glob",
                "-i", f"{acc_render_dir}/*.png",
                "-frames:v", str(params.inference.rollout.n_frames),
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
            # delete all png files in render directory
            for file in os.listdir(acc_render_dir):
                if file.endswith(".png"):
                    os.remove(os.path.join(acc_render_dir, file))


if __name__ == '__main__':
    params.wandb.log = False    # disable wandb logging
    params.training = False
    device = 'cuda' if params.inference.cuda else 'cpu'

    cuda = True if device == 'cuda' else False
    print('device:', device)
    print('cuda:', cuda)
    if params.inference.renderer.lower() == "nvdiffrast":
        import nvdiffrast.torch as dr
    main()