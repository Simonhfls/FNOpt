import logging
import os
from pathlib import Path
import numpy as np
import torch
from configs.config_common import motion_presets
from get_param2 import params
from sft import evaluation
from sft.utils import loadJson

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    simulation_frames = params.inference.rollout.n_frames
    device = torch.device(params.inference.device)
    scene_list = params.inference.rollout.json
    motion_code_list = motion_presets[params.inference.rollout.input_data.motion_code] if params.inference.rollout.using_handle_traj else [None]
    task_list = [(file_name, motion_code) for file_name in scene_list for motion_code in motion_code_list]
    input_data = dict(params.inference.rollout.input_data)
    scale_cd = 1e3
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

    # for file_name in scene_list:
    for file_name, motion in task_list:
        do_rollout = params.inference.rollout.save_npy
        save_npy = params.inference.rollout.save_npy
        print(f'file_name: {file_name}, motion: {motion}' )
        scene_parameters = loadJson(file_name)
        input_data['motion_code'] = motion

        # Load the mesh
        mode = input_data['mode']
        npy_file_name = f"{input_data['obj_code']}_{motion}_e_800.npy"

        save_dir = f"evaluation/S_MGNRP/npy_results/{params.net.name.lower()}/{mode}/"

        save_path_npy = os.path.join(save_dir, npy_file_name)
        print(f'npy file path: {os.path.abspath(save_path_npy)}')

        try:
            npy_results = np.load(save_path_npy, allow_pickle=True).item()
        except FileNotFoundError:
            print(f"File not found: {os.path.abspath(save_path_npy)}. Skipping evaluation.")
        else:
            print(f"Loaded results from {os.path.abspath(save_path_npy)}")

        log_path = Path(f"evaluation/S_MGNRP/eval_log_{params.net.name}.txt")
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
        logging.info('-' * 80)



        # load ground truth point clouds
        ground_truth_point_clouds, point_clouds_lengths = (
            evaluation.loadGroundTruthMGNRP(
            scene_parameters, input_data, device=device
        ))
        our_point_clouds = torch.zeros_like(ground_truth_point_clouds)
        positions = torch.from_numpy(npy_results['position']).to(device)
        faces = torch.from_numpy(npy_results['face']).to(device)

        assert positions.shape[0] == ground_truth_point_clouds.shape[0], \
            f"Mismatch in number of frames: {positions.shape[0]} vs {ground_truth_point_clouds.shape[0]}"

        for i in range(ground_truth_point_clouds.shape[0]):
            our_point_clouds[i] = evaluation.sampleMesh(point_clouds_lengths[i],
                                                        positions[i],
                                                        faces,
                                                        device=device)

        # debug save pcl
        # evaluation.savePointCloud(our_point_clouds[700], f'./pcl_ours_{motion}.obj')
        # evaluation.savePointCloud(ground_truth_point_clouds[700], f'./pcl_gt_{motion}.obj')

        chamfer_distance = evaluation.computeChamferDistance(ground_truth_point_clouds,
                                                             our_point_clouds,
                                                             0,
                                                             len(ground_truth_point_clouds),
                                                             point_clouds_lengths,
                                                             inverse=False,
                                                             scale=scale_cd)

        logging.info(
            f"Model: {params.net.name}{params.inference.postfix} | "
            f"Iter/ts: {params.inference.iterations_per_timestep:<3} | "
            f"Motion: {motion:<12} | "
            f"Wind: {params.inference.rollout.wind_density:<3} | "
            f"Chamfer(s={scale_cd}): {chamfer_distance:.4f}"
        )





