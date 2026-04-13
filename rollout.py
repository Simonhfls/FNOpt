import math
import os
import time
import uuid
from pathlib import Path
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LightSource
from pytorch3d.io import load_obj, save_obj
from pytorch3d.transforms import axis_angle_to_matrix
from Logger import Logger
from compute_forces import compute_wind_force
from dataset_cloth import DatasetCloth
from dataset_utils import DatasetToSingleChannel
from fnopt import get_Net
from generate_json_conf import setup_handle_traj, get_path_from_gt_input, setup_handle_traj_gt, \
    obj_name_to_handle_ind_list_dict, change_sequence_speed
from get_param import toCuda, get_hyperparam, device, update_params_inference
from losses.repulsive_energy import RepulsiveEnergy
from preprocess import transform_positions, generate_correction_traj
import evaluation
from utils import get_face_areas_batch, get_f_connectivity_edges
from get_param import params

params = update_params_inference(params)
params.wandb.log = False
params.training = False

repulsive_loss = RepulsiveEnergy(threshold=params.cloth.repulsive.thres)

class Rollout():
    def __init__(self, evaluate=False, device='cuda'):
        self.device = device
        self.dtype = params.net.dtype
        self.evaluate = evaluate

    def initializeParameters(self, scene, input_data=None):
        self.scene_parameters = scene
        if 'mesh_resolution' in scene:
            self.mesh_resolution = scene['mesh_resolution']
            self.h = self.mesh_resolution
            self.w = self.mesh_resolution
        else:
            self.h = scene['mesh_resolution_w']  # inverse order because of different interpretation
            self.w = scene['mesh_resolution_h']
            self.mesh_resolution = f'{self.h}x{self.w}'
        if 'mesh_size_h' in scene and 'mesh_size_w' in scene:
            self.mesh_size_h = scene['mesh_size_h']
            self.mesh_size_w = scene['mesh_size_w']
        else:
            self.mesh_size_h = scene['mesh_size']
            self.mesh_size_w = scene['mesh_size']
        self.mesh_area = self.mesh_size_h * self.mesh_size_w
        self.dir_dataset = scene['dataset_dir']
        self.input_data = input_data
        self.abl_optimizers = (params.name == 'abl_optimizers')
        self.frame_counter = 0
        self.t_iter = 0
        self.epoch_counter = 0
        self.dpi = 200
        self.speed_factor = None
        self.time_conversion = 50
        self.length_conversion = (self.h - 1) / self.mesh_size_w
        self.bc_n_x = math.ceil(self.h / 32)
        self.unique_id = uuid.uuid4().hex[:8]
        self.dir_rgb = os.path.abspath(os.path.join('evaluation', self.scene_parameters['scene'], 'rendered_rollout', 'tmp' + self.unique_id))
        if params.inference.rollout.save_render_fnopt and not os.path.exists(self.dir_rgb):
            os.makedirs(self.dir_rgb)

        self.dt = params.cloth.dt

        self.renderer = params.inference.renderer.lower()
        print('renderer:', self.renderer)
        self.M_repulsive_list = []      # a list of length self.simulation_frames, each element is a float number

    def initializeMesh(self):
        verts, faces, aux = load_obj(str(Path(self.dir_dataset) / self.scene_parameters["mesh_file"]), load_textures=True, device=self.device)
        self.rest_positions = verts.to(dtype=self.dtype)
        self.faces = faces.verts_idx.to(dtype=torch.int32)
        self.f_connectivity_edges = get_f_connectivity_edges(self.faces)
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

        self.rest_positions = self.rest_positions * self.length_conversion
        self.rest_positions = self.rest_positions.clone()
        self.original_uv = self.uv.clone()
        self.uv = self.uv.clone()
        self.positions_net = self.rest_positions.transpose(0, 1).view(3, self.h, self.w).type(self.dtype)
        velocities_net = torch.zeros(3, self.h, self.w, device=self.device)
        self.x_v = torch.cat([self.positions_net, velocities_net]).unsqueeze(0)
        self.bc_indices = None

        if self.input_data and self.input_data['motion_code']:
            handle_ind_list = self.scene_parameters["handle_ind_list"]
            self.handle_mask = torch.ones_like(self.rest_positions).bool()
            for handle_ind in handle_ind_list:
                self.handle_mask[handle_ind, :] = False

            mask_3hw = self.handle_mask.permute(1, 0).reshape(3, self.h, self.w)
            non_handle_grid = mask_3hw.any(dim=0)  # (H,W)
            handle_grid = ~non_handle_grid  # (H,W) True=handle
            rc = torch.nonzero(handle_grid, as_tuple=False)  # (K,2)
            self.bc_indices = rc.tolist()

            # load mgnrp mesh
            obj_path = str(Path(self.dir_dataset) / self.scene_parameters["mesh_file"])
            verts, _, _ = load_obj(obj_path, load_textures=True, device=self.device)

            if self.input_data['mode'] == 'arbitrary':
                handle_traj = setup_handle_traj(verts, self.input_data['motion_code'], list(reversed(handle_ind_list)),params.inference.rollout.framerate) # (F, V, 3)

            elif self.input_data['mode'] == 'with_gt':
                gt_path = get_path_from_gt_input(self.input_data, os.path.join(self.scene_parameters['gt_data_dir'], 'gt_data'))

                handle_traj_mgn = setup_handle_traj_gt(gt_path, obj_name_to_handle_ind_list_dict[self.input_data['obj_code']])
                handle_traj = torch.zeros((handle_traj_mgn.shape[0], self.h * self.w, 3), device=self.device, dtype=self.dtype)
                handle_traj[:, list(reversed(handle_ind_list))] = handle_traj_mgn[:, obj_name_to_handle_ind_list_dict[self.input_data['obj_code']][:len(handle_ind_list)]]
                #extend the first frame to match gt length
                handle_traj = torch.cat([handle_traj, handle_traj[-1:]], dim=0) # todo check

            if 'speed' in self.input_data and self.input_data['speed'] != 1.0:
                self.speed_factor = float(self.input_data['speed'])
                if self.input_data['mode'] == 'arbitrary':
                    hold_frames = 300
                else:
                    hold_frames = 0
                handle_traj = change_sequence_speed(handle_traj, self.speed_factor, hold_frames=hold_frames)

            scale_ratio = (self.h - 1) / self.mesh_size_w    # to take into account flat mesh case
            handle_disp = handle_traj - torch.where(self.handle_mask, handle_traj, handle_traj[0])
            self.R12 = rotation_matrix.unsqueeze(0).repeat(handle_traj.shape[0], 1, 1)
            self.T12 = torch.tensor([0., 0., 0.], device=self.device).unsqueeze(0).repeat(handle_traj.shape[0], 1, 1)
            handle_disp = transform_positions(handle_disp.permute(0, 2, 1), self.R12, self.T12)

            first_frame_mesh_pos = transform_positions(handle_traj.permute(0, 2, 1) * self.length_conversion, self.R12, self.T12)[0]  # (H*W, 3)
            self.handle_traj = torch.where(self.handle_mask, torch.zeros_like(self.rest_positions), first_frame_mesh_pos) + handle_disp * scale_ratio
            self.handle_traj = self.handle_traj.permute(0, 2, 1).reshape(self.handle_traj.shape[0], 3, self.h, self.w).to(self.device)

            handle_mask_flat = ~self.handle_mask.any(dim=-1)
            rest_handle = self.rest_positions[handle_mask_flat]  # (n_handle, 3)
            traj0_handle = self.handle_traj[0].permute(1, 2, 0).reshape(-1, 3)[handle_mask_flat]  # (n_handle, 3)

            dist = (rest_handle - traj0_handle).norm(dim=-1).mean()
            if dist > 0.1:
                # if initial position doesn't align, first gradually move handle_traj to mgnrp sequence's initial position
                print(f"Initial handle position does not align with rest positions. Distance: {dist:.3f}")
                print("-> Applying initial correction trajectory to handle positions...")
                initial_correction_traj = generate_correction_traj(target=self.handle_traj[0],
                                                                   source=self.rest_positions,
                                                                   mask=self.handle_mask,
                                                                   n_steps=200,
                                                                   device=self.device)    # todo how to best generate this trajectory?
                self.handle_traj = torch.cat([initial_correction_traj, self.handle_traj], dim=0)
                self.R12 = rotation_matrix.unsqueeze(0).repeat(self.handle_traj.shape[0], 1, 1)
                self.T12 = torch.tensor([0., 0., 0.], device=self.device).unsqueeze(0).repeat(self.handle_traj.shape[0], 1, 1)


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
        if not self.abl_optimizers:
            network = toCuda(get_Net(params))
            logger = Logger(get_hyperparam(params), use_csv=False, use_tensorboard=False)
            print('load_date_time:', params.inference.load_date_time)
            date_time, index = logger.load_state(network, None, datetime=params.inference.load_date_time, index=params.inference.load_index, device=self.device)
            print(f"loaded: {date_time}, {index}")
            self.load_index = index
            self.cloth_net = network
            self.cloth_net.eval()
        else:
            self.load_index = -1
            self.opt_type = params.net.name
            self.lr = params.optimizer.lr
            self.beta1, self.beta2 = 0.9, 0.999  # for Adam

            # for LBFGS
            self.lbfgs_hist = params.optimizer.lbfgs_hist
            self.lbfgs_step = self.lr
            self.lbfgs_eps = params.optimizer.lbfgs_eps
            self.max_step_norm = params.optimizer.max_step_norm

            self.eps = 1e-8
            self.opt_state = {}

            self.opt_state['default'] = {
                "m": torch.zeros((3, 1, self.h, self.w)).to(device),  # Adam m
                "v_hat": torch.zeros((3, 1, self.h, self.w)).to(device),  # Adam v
                "lbfgs": {
                    "s_list": [],
                    "y_list": [],
                    "prev_grad": None,
                    "prev_update": None,
                    "m": getattr(self, "lbfgs_hist", 10)
                }
            }

    def initializeOptimization(self):
        gravity = torch.tensor(self.scene_parameters["gravity"], device=self.device, dtype=torch.float32)
        self.external_forces = torch.tensor(gravity * self.length_conversion / (self.time_conversion * self.time_conversion), device=self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(2).unsqueeze(3)   # todo check
        self.vertex_forces = torch.zeros((1, self.simulation_frames, 3, self.h, self.w), device=self.device, dtype=torch.float32)

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

        self.original_dataset.reset0_inference_env(self.positions_net, bc_indices=self.bc_indices)
        self.original_dataset.set_optimizable(self.external_forces, self.stretching_stiffness, self.shearing_stiffness, self.bending_stiffness)
        if self.input_data and self.input_data['motion_code']:
            self.original_dataset.set_bc_positions(self.handle_traj[self.frame_counter])

    def initialize(self, scene, input_data=None):
        print("Start: Initialization")
        t_start = time.perf_counter()

        self.initializeParameters(scene, input_data)
        self.initializeMesh()
        self.initializeNetwork()
        self.initializeOptimization()

        if self.evaluate:
            ground_truth_point_clouds, point_clouds_lengths = evaluation.loadGroundTruthMGNRP(
                self.scene_parameters, self.input_data, device=self.device
            )

            self.point_clouds = {"ground_truth": ground_truth_point_clouds,
                                 "lengths": point_clouds_lengths}

        
        self.cloth_m = np.load(os.path.join(self.scene_parameters['gt_data_dir'], 'unity_demo', 'square_1024', 'cloth_m.npy'))

        self.mass_conversion = np.sum(self.cloth_m) * self.mesh_area / (self.h * self.w)  # average mass per vertex
        self.real_mass_vertices = (self.original_dataset.M / (self.original_dataset.M.sum()) * self.cloth_m.sum()).reshape(-1, 1)

        t_end = time.perf_counter()
        print(f"Done:  Initialization in {t_end - t_start:.3f} s\n")

    def step(self):
        a_ext = self.external_forces + self.vertex_forces[:, self.frame_counter]
        # wind
        if params.inference.rollout.wind_density > 0.0:
            a_ext = a_ext + self.wind_force
        last_iter = ((self.t_iter + 1) % params.inference.iterations_per_timestep == 0)
        self.original_dataset.set_optimizable(a_ext, self.stretching_stiffness, self.shearing_stiffness, self.bending_stiffness)
        grads, hidden_states = self.test_dataset.ask_inference(retain_graph=False)
        if params.name == 'abl_optimizers':
            update_steps, new_hidden_states = self.step_classical_opt(grads, hidden_states, last_iter)
            self.scales.append(-1)
        else:
            update_steps, new_hidden_states = self.step_fnopt(grads, hidden_states)
            self.scales.append(new_hidden_states[0][2][0, 0, 0, 0].item())

        bc_vel = None
        if self.input_data and self.input_data['motion_code']:
            bc_vel = self.bc_vel_list[self.frame_counter]
            self.original_dataset.set_bc_positions(self.handle_traj[self.frame_counter])

        _ = self.test_dataset.tell_inference(update_steps, new_hidden_states, detach_acc=True, bc_velocity=bc_vel)
        self.gradients.append(torch.norm(grads, p=2).item())

        if last_iter:
            # update wind force
            cloth_v = self.original_dataset.v.permute(0, 2, 3, 1).reshape(-1, 3) * self.time_conversion / self.length_conversion  # convert to m/s
            cloth_f_area = get_face_areas_batch(vertices=self.original_dataset.x.permute(0, 2, 3, 1).reshape(1, -1, 3), faces=self.faces) / (self.length_conversion ** 2)  # compute face area
            cloth_pos = self.original_dataset.x.permute(0, 2, 3, 1).reshape(-1, 3) / self.length_conversion  # convert to m
            face_tensor = self.faces

            wind_force = compute_wind_force(self.original_dataset.M.squeeze().reshape(-1, 1),
                                            cloth_f_area.permute(1, 0),      # real
                                            face_tensor,
                                            cloth_v,        # real
                                            cloth_pos,      # real
                                            wind_density=params.inference.rollout.wind_density)
            # wind_force = wind_force * self.length_conversion / (self.time_conversion * self.time_conversion * self.mass_conversion)
            wind_force = wind_force * self.length_conversion / (self.time_conversion * self.time_conversion * self.real_mass_vertices)
            self.wind_force = wind_force.reshape(1, self.h, self.w, 3).permute(0, 3, 1, 2)
            self.frame_counter += 1
            if self.frame_counter % 50 == 0:
                print(
                    f'current frame: {self.frame_counter:>4} | '
                    f'input grads: {self.gradients[-1]:.3f} | '
                    f'step scale: {self.scales[-1]:.3f} | '
                    f'update step: {float(torch.norm(update_steps, p=2)):.3f}'
                )
            index = 0
            x = self.original_dataset.x[index]
            predicted_pos = x.view(3, -1).transpose(0, 1) / self.length_conversion
            self.predicted_pos[self.frame_counter] = predicted_pos.squeeze()

            if params.inference.rollout.log_repulsive:
                # log metrics
                M_repulsive = repulsive_loss(x.view(3, -1).transpose(0, 1).unsqueeze(0), self.f_connectivity_edges)
                self.M_repulsive_list.append(M_repulsive.item())
                print(f"M_repulsive: {M_repulsive.item():.6f}")

            ### Visualization
            if params.inference.visualize_3d:  # visualize 3D cloth
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

                ax.set_zlim(-params.inference.height * 1.2, 1.01)
                ax.set_xlim(-params.inference.height * 0.6, params.inference.height * 0.6)
                ax.set_ylim(-params.inference.height * 0.6, params.inference.height * 0.6)
                plt.title(f"timestep: {self.original_dataset.T[index].cpu().numpy()[0]}")


                self.path_metamizer = f"plots/{get_hyperparam(params).replace(' ', '_').replace(';', '_')}/cloth/{params.inference.load_date_time}/stiff_{params.inference.material.stretching} shear_{params.inference.material.shearing} bend_{params.inference.material.bending} iters_{params.inference.iterations_per_timestep}/tmp{self.unique_id}"
                os.makedirs(self.path_metamizer, exist_ok=True)
                assert os.path.exists(self.path_metamizer), f"Failed to create path: {self.path_metamizer}"
                # plt.savefig(f"{self.path_metamizer}/frame_{str(self.frame_counter).zfill(4)}.png", dpi=self.dpi)

                plt.draw()
                plt.pause(0.01)

                if params.inference.save_obj and self.frame_counter % 20 == 0:
                    # save mesh
                    mesh_path = f"{self.path_metamizer}/frame_{str(self.frame_counter).zfill(4)}.obj"
                    print('mesh_path:', os.path.abspath(mesh_path))
                    save_obj(mesh_path, x.permute(1, 2, 0).reshape(-1, 3) / self.length_conversion, self.faces)

            if params.inference.visualize_scaling:  # visualize, how scaling changes during update steps
                plt.figure(2)
                plt.clf()
                stride = 1  # len(scales)//200+1
                plt.semilogy(self.scales[::stride])
                plt.xlabel("iteration")
                plt.ylabel("scale")
                plt.legend(["scales"])
                plt.title(f"Scale, {params.inference.iterations_per_timestep} iterations per timestep, {self.h} x {self.w}")

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
                    f"Gradient Norm, {params.inference.iterations_per_timestep} iterations per timestep, {self.h} x {self.w}")
                plt.draw()
                plt.pause(0.01)

        self.t_iter += 1


    def step_fnopt(self, grads, hidden_states):
        return self.cloth_net(grads, hidden_states)

    def step_classical_opt(self, grads, hidden_states, last_iter):
        key = "default"
        opt = self.opt_state[key]
        new_hidden_states = None

        if self.opt_type == "adam":
            # --- Adam update
            opt["m"] = self.beta1 * opt["m"] + (1 - self.beta1) * grads
            opt["v_hat"] = self.beta2 * opt["v_hat"] + (1 - self.beta2) * grads.pow(2)
            m_hat = opt["m"] / (1 - self.beta1)
            v_hat = opt["v_hat"] / (1 - self.beta2)
            update_steps = -self.lr * m_hat / (v_hat.sqrt() + self.eps)

        elif self.opt_type == "lbfgs":
            state = opt["lbfgs"]

            if state["prev_grad"] is not None and state["prev_update"] is not None:
                y = grads - state["prev_grad"]  # y_{k-1} = g_k - g_{k-1}
                s = state["prev_update"]  # s_{k-1} = x_k - x_{k-1}
                ys = torch.sum(y * s)

                if torch.isfinite(ys) and ys.item() > self.lbfgs_eps:
                    if len(state["s_list"]) >= state["m"]:
                        state["s_list"].pop(0)
                        state["y_list"].pop(0)
                    state["s_list"].append(s.detach())
                    state["y_list"].append(y.detach())


            q = grads.detach().clone()
            alphas = []

            # round one: from near to far
            for s_i, y_i in zip(reversed(state["s_list"]), reversed(state["y_list"])):
                rho_i = 1.0 / (torch.sum(y_i * s_i) + self.lbfgs_eps)
                alpha_i = rho_i * torch.sum(s_i * q)
                q = q - alpha_i * y_i
                alphas.append(alpha_i)


            if len(state["s_list"]) > 0:
                s_last, y_last = state["s_list"][-1], state["y_list"][-1]
                gamma = torch.sum(s_last * y_last) / (torch.sum(y_last * y_last) + self.lbfgs_eps)
            else:
                gamma = torch.tensor(1.0, device=grads.device, dtype=grads.dtype)

            r = gamma * q

            # round 2: from far to near
            for i in range(len(state["s_list"])):
                s_i, y_i = state["s_list"][i], state["y_list"][i]
                rho_i = 1.0 / (torch.sum(y_i * s_i) + self.lbfgs_eps)
                beta_i = rho_i * torch.sum(y_i * r)
                alpha_i = alphas[-(i + 1)]
                r = r + s_i * (alpha_i - beta_i)

            direction = -r

            # ---- (C) step size and update steps -------------------
            step_size = self.lbfgs_step  # constant step size
            update_steps = step_size * direction

            if self.max_step_norm is not None:
                # optional: limit the step norm
                step_norm = update_steps.norm()
                if torch.isfinite(step_norm) and step_norm.item() > self.max_step_norm:
                    update_steps = update_steps * (self.max_step_norm / (step_norm + 1e-12))

            # ---- (D) record current state for next iteration ------
            state["prev_grad"] = grads.detach()
            state["prev_update"] = update_steps.detach()

            # if it's the last iteration of the current timestep, clear the state
            if last_iter:
                state["s_list"].clear()
                state["y_list"].clear()
                state["prev_grad"] = None
                state["prev_update"] = None

        elif self.opt_type == "gd":  # "gd"
            update_steps = -self.lr * grads

        return update_steps, new_hidden_states

    def save_acc_visu(self):
        a_all = self.predicted_a.cpu().detach().numpy()
        a_all = np.clip(a_all, np.percentile(a_all, 5), np.percentile(a_all, 95))
        vmin_global = np.min(a_all)
        vmax_global = np.max(a_all)
        self.dir_acc = os.path.abspath(os.path.join('evaluation', self.scene_parameters['scene'], 'rendered_acc', 'tmp' + self.unique_id))
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
