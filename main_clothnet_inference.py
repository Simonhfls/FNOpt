import os
import sys
import logging
import traceback
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from render_utils import setup_evaluation_logger, check_frame, generate_heatmap_video, try_acquire_lock, release_lock, \
    render_single, render_video_with_gt
from archive.rollout import Rollout
from tri_to_quad_mesh import batch_trimesh_to_quadmesh_torch
from motion_codes import motion_presets
from generate_json_conf import get_path_from_gt_input
from grid_mesh import GridMesh
from preprocess import transform_positions
from tri_to_quad_mesh import load_obj_with_uv
from get_param import params, update_params_inference
import evaluation
from utils import loadJson
import numpy as np
import time
import matplotlib.pyplot as plt
import torch


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scene_list = params.inference.rollout.json
    motion_codes = motion_presets[params.inference.rollout.input_data.motion_code] if params.inference.rollout.using_handle_traj else [None]
    print('motion_code:', params.inference.rollout.input_data.motion_code)
    print('using_handle_traj', params.inference.rollout.using_handle_traj)
    print('motion_codes:', motion_codes)
    task_list = [(file_name, motion_code) for file_name in scene_list for motion_code in motion_codes]
    print('\n\n\n**************************')
    print('device:', device)
    print('model:', params.net.name)
    print('renderer:', params.inference.renderer.lower())
    print('N_iter_per_step:', params.inference.iterations_per_timestep)
    print('dtype:', params.net.dtype)
    if params.inference.rollout.using_handle_traj:
        print('motion codes:', motion_codes)
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
        opt = Rollout(evaluate=evaluate, device=device)
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
        epoch = 800 if params.net.name == 'MGNRP' else opt.load_index
        res = opt.mesh_resolution if 'mesh_resolution' in opt.__dict__ else scene['mesh_resolution']
        mode = input_data['mode']
        save_dir = f"evaluation/S_MGNRP/npy_results/{params.net.name}{params.inference.postfix}/{mode}/"
        if params.name == 'abl_optimizers':
            result_filename_base = f"{input_data['obj_code']}_{input_data['motion_code']}_e_{epoch}_Y{Y}_S{S}_B{B}_WD{WD}_iter{iter}_res{res}_abl_optimizer_{opt.opt_type}_lr_{opt.lr}"
        elif params.net.name == 'MGNRP':
            # result_filename_base = f"template_mgnrp_quad_{res}_{input_data['motion_code']}_e_{epoch}"
            result_filename_base = f"{opt.scene_parameters['mesh_file'].rpartition('_')[0]}_{res}_{input_data['motion_code']}_e_{epoch}"
            if opt.speed_factor is not None:
                result_filename_base += f'_speed{opt.speed_factor}'
            if 'bc_desc' in opt.scene_parameters:
                result_filename_base = f"template_mgnrp_quad_{res}_bc_{opt.scene_parameters['bc_desc'].replace('-', '_')}_{input_data['motion_code']}_e_{epoch}"
        else:
            repulsive_str = f"_R{params.cloth.repulsive.k}_{params.cloth.repulsive.thres}" if params.cloth.repulsive.k > 0 else ""

            result_filename_base = f"{input_data['obj_code']}_{input_data['motion_code']}_e_{epoch}_Y{Y}_S{S}_B{B}_WD{WD}{repulsive_str}_iter{iter}_res{res}"
            if opt.speed_factor is not None:
                result_filename_base += f'_speed{opt.speed_factor}'
            if 'bc_desc' in opt.scene_parameters:
                result_filename_base += f"_bc{opt.scene_parameters['bc_desc']}"
        npy_filename = result_filename_base + '.npy'
        save_path_npy = os.path.join(save_dir, npy_filename)
        print(f'npy file path: {os.path.abspath(save_path_npy)}')

        if evaluate:
            setup_evaluation_logger(
                evaluate=evaluate,
                model_name=params.net.name,
                resolution=opt.mesh_resolution,
                postfix=params.inference.postfix
            )

        do_rollout = save_npy or not params.inference.rollout.using_handle_traj
        if motion and save_npy:
            os.makedirs(save_dir, exist_ok=True)
            if os.path.exists(save_path_npy):
                print(f"[Skip] Already exists: {os.path.abspath(save_path_npy)}")
                do_rollout = False
                save_npy = False
            else:
                if not try_acquire_lock(save_path_npy):
                    print(f"[Skip] Another process is predicting (lock present): {os.path.abspath(save_path_npy)}.lock")
                    do_rollout = False
                    save_npy = False

        time1 = time.perf_counter()

        per_frame_duration = 0.0
        if do_rollout:
            try:
                while (opt.t_iter < opt.simulation_frames * params.inference.iterations_per_timestep):
                    opt.step()
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
                    f"Res: {res:<3} | "
                    f"{type(e).__name__}: {str(e)}"
                )
                traceback.print_exc()
                continue
            else:
                release_lock(save_path_npy)
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
            if len(opt.M_repulsive_list) > 0:
                save_dict['M_repulsive_framewise'] = np.array(opt.M_repulsive_list)

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
            gt_path = get_path_from_gt_input(input_data, os.path.join(scene['gt_data_dir'], 'gt_data'))
            gt_x = torch.from_numpy(np.load(os.path.join(gt_path, "cloth_pos.npy"))).to(device)
            _, gt_f, gt_uv = load_obj_with_uv(os.path.join(scene['gt_data_dir'], 'unity_demo', 'square_1024.obj'))

            ground_truth_point_clouds = opt.point_clouds["ground_truth"]
            our_point_clouds = torch.zeros_like(ground_truth_point_clouds)
            faces = torch.from_numpy(npy_results['face']).to(device)

            assert predicted_pos.shape[0] == ground_truth_point_clouds.shape[0], \
                f"Mismatch in number of frames: {predicted_pos.shape[0]} vs {ground_truth_point_clouds.shape[0]}"
            try:
                print('Evaluating e3D...')
                gt_f = torch.from_numpy(gt_f).to(device)
                gt_uv = torch.from_numpy(gt_uv).to(device)
                quad_verts, quad_quads = batch_trimesh_to_quadmesh_torch(gt_x, gt_f, gt_uv, (opt.h, opt.w))
                quad_verts = quad_verts.reshape(quad_verts.shape[0], -1, 3)
                e3d = evaluation.computeMeshDistance(quad_verts, predicted_pos, 0, len(gt_x))

                # compute chamfer distance
                for i in range(ground_truth_point_clouds.shape[0]):
                    pos = predicted_pos[i]
                    check_frame(pos, faces, i)
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
                f"Model: {params.net.name}{params.inference.postfix}(e{opt.load_index}) | "
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

        if params.inference.rollout.log_repulsive:
            print('computing repulsive metrics...')
            if 'M_repulsive_framewise' not in npy_results:
                print("No repulsive constraint data found in the results.")
            else:
                M_repulsive_framewise = npy_results['M_repulsive_framewise']
                print(f"Average number of repulsive constraints: {np.mean(M_repulsive_framewise):.2f} ± {np.std(M_repulsive_framewise):.2f}")
                # define several thresholds for computing percentage of frames whose l_repulsive is above a certain threshold
                thresholds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0, 1000.0]
                for threshold in thresholds:
                    percentage = np.mean(M_repulsive_framewise > threshold) * 100
                    print(f"Percentage of frames with M_repulsive > {threshold}: {percentage:.2f}%")

        if params.inference.rollout.save_acc_heatmap:
            opt.save_acc_visu()
            print(f"Rendering accuracy heatmaps into {opt.dir_acc}")
            template = (
                f"V_ACC_{params.net.name}{params.inference.postfix}_"
                f"RES{opt.mesh_resolution}_Y{params.inference.material.stretching}_"
                f"S{params.inference.material.shearing}_B{params.inference.material.bending}_"
                f"EP{opt.load_index}_FPS{params.inference.rollout.framerate}.mp4"
            )
            generate_heatmap_video(
                render_dir=opt.dir_acc,
                filename_template=template,
                fps=params.inference.rollout.framerate,
                n_frames=params.inference.rollout.n_frames
            )

        mp4_file = result_filename_base + '.mp4'
        if params.inference.rollout.save_render_fnopt:
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
                if not try_acquire_lock(save_path):
                    print(f"[Skip] Another process is rendering (lock present): {save_path}.lock")
                    skip = True

            if not skip:
                try:
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
                            gt_path = get_path_from_gt_input(input_data, os.path.join(scene['gt_data_dir'], 'gt_data'))
                            if gt_x is None and opt.speed_factor is None:
                                gt_x = torch.from_numpy(np.load(os.path.join(gt_path, "cloth_pos.npy"))).to(device)
                            R12 = opt.R12[0].unsqueeze(0).repeat(gt_x.shape[0], 1, 1)
                            T12 = opt.T12[0].unsqueeze(0).repeat(gt_x.shape[0], 1, 1)
                            gt_x = transform_positions(gt_x.permute(0, 2, 1), R12, T12)
                            _, gt_f, gt_uv = load_obj_with_uv(os.path.join(scene['gt_data_dir'], 'unity_demo', 'square_1024.obj'))
                            gt_uv = torch.from_numpy(gt_uv).to(device)
                            gt_f = torch.from_numpy(gt_f).to(device)


                    render_video_with_gt(
                        x_preds=torch.from_numpy(x_list).to(device),  # (F,3,H,W)
                        gt_vertices_list=gt_x,  # (F,V,3)
                        gt_faces=gt_f,  # (n_faces,3)
                        gt_uv=gt_uv,
                        bc_mask_list=bc_mask_list,
                        save_path=save_path,
                        fps=params.inference.rollout.framerate,
                        gt_alpha=gt_alpha,
                        pred_alpha=pred_alpha,
                        azim=-60,       #-60
                        elev=30,        #30
                        dpi=800,            # 800
                        debug=params.inference.visualize_3d,
                        show_handle_points=params.inference.rollout.show_handle_points,
                        show_error_map=plot_error_map,
                    )
                    print(f"Render video saved to: {os.path.abspath(save_path)}")
                finally:
                    release_lock(save_path)

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
                    gt_path = get_path_from_gt_input(input_data, os.path.join(scene['gt_data_dir'], 'gt_data'))
                    gt_x = np.load(os.path.join(gt_path, "cloth_pos.npy"))
                    _, f, _ = load_obj_with_uv(os.path.join(scene['gt_data_dir'], 'unity_demo', 'square_1024.obj'))
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
            plt.title(f"Gradient Norm, {params.inference.iterations_per_timestep} iterations per timestep, {opt.h} x {opt.w}")
            plt.draw()
            plt.savefig(f"{render_dir}/V_GRADS_{params.net.name}{params.inference.postfix}_RES{opt.mesh_resolution}_Y{params.inference.material.stretching}_S{params.inference.material.shearing}_B{params.inference.material.bending}_EP{opt.load_index}_iters{params.inference.iterations_per_timestep}.png", dpi=dpi)
            pass

if __name__ == '__main__':
    params = update_params_inference(params)
    params.wandb.log = False
    params.training = False
    print("torch.get_num_threads():", torch.get_num_threads())

    main()