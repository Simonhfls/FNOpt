import logging
from pathlib import Path
import numpy as np
from pytorch3d.io import save_obj
from configs.config_common import motion_presets
from sft.evaluation import sampleMesh, computeChamferDistance
from sft.utils import save_pcl
# allow command line arguments
import argparse

"""
This script evaluates the MGNRP model using Chamfer distance.
"""

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_points", type=int, default=10000, help="number of points to sample from the mesh")
    parser.add_argument("--mgnrp_root_dir", type=str, default='/Users/ruochen/Documents/liris_code/meshgraphnet_rp/', help="root directory of MGNRP")
    parser.add_argument("--resolution", type=int, default=24, help="rollout resolution to compute chamfer distance")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt = get_args()
    motion_codes_eval = motion_presets['eval']
    resolution = opt.resolution  # resolution of the mesh, used for naming the output files
    print('resolution:', resolution)
    # logging
    log_path = Path(f"evaluation/S_MGNRP/mgnrp_log{resolution}.txt")
    print(f'logging save to: {log_path}')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode='a'),  # append to log file
            logging.StreamHandler()  # print to console
        ]
    )
    logging.info("*****Starting evaluation of MGNRP with Chamfer distance*****")
    N = opt.n_points  # number of points to sample from the mesh
    for motion in motion_codes_eval:
        # motion = 'yz_v2'
        # npy_file = f"square_{resolution}_{motion}_e_800.npy"
        npy_file = f"template_mgnrp_quad_{resolution}_{motion}_e_800.npy"
        npy_path_pred = Path(opt.mgnrp_root_dir) / f"output/npy_results/final_model/with_gt/{npy_file}"

        # load npy data
        npy_data_pred = np.load(npy_path_pred, allow_pickle=True).item()
        position_pred = torch.from_numpy(npy_data_pred['position']).to(device)
        face = torch.from_numpy(npy_data_pred['face']).to(device)
        position_pred = position_pred[1:]  # skip the first frame
        points_predicted = torch.zeros((position_pred.shape[0], N, 3)).to(device)
        for i in range(position_pred.shape[0]):
            points_predicted[i] = sampleMesh(N, position_pred[i], face, device=device)

        # gt
        npy_path_gt = Path(opt.mgnrp_root_dir) / f"input/gt_data/square_1024_{motion}/cloth_pos.npy"
        position_gt = np.load(npy_path_gt, allow_pickle=True)
        position_gt = torch.from_numpy(position_gt).to(device)
        points_gt = torch.zeros((position_gt.shape[0], N, 3)).to(device)
        # face_gt0 = torch.from_numpy(np.load(opt.mgnrp_root_dir + 'input/gt_data/square_1024_yz_v2_opp/face.npy')).to(device)
        face_path = Path(opt.mgnrp_root_dir) / 'input' / 'gt_data' / 'square_1024_yz_v2_opp' / 'face.npy'
        face_gt = torch.from_numpy(np.load(face_path)).to(device)

        for i in range(position_gt.shape[0]):
            points_gt[i] = sampleMesh(N, position_gt[i], face_gt, device=device)

        # debug save pcl
        save_pcl(points_predicted[2], f'pred_debug.obj')
        save_pcl(points_gt[2], f'gt_debug.obj')

        # debug save mesh
        for i in range(points_gt.shape[0]):
            if not Path('debug').exists():
                Path('debug').mkdir(parents=True, exist_ok=True)
            save_obj(f'debug/pred_mesh_{i}.obj', position_pred[i], face)
            save_obj(f'debug/gt_mesh_{i}.obj', position_gt[i], face_gt)
        # save_obj(f'pred_mesh_debug.obj', position_pred[41], face)
        # save_obj(f'gt_mesh_debug.obj', position_gt[40], face_gt)

        # log chamfer distance
        assert points_predicted.shape[0] == points_gt.shape[0]
        n_frames = points_predicted.shape[0]
        pcl_lengths = torch.tensor([N] * n_frames, dtype=torch.int64, device=device)
        chamfer_distance = computeChamferDistance(points_predicted, points_gt, 0, n_frames, pcl_lengths, scale=1,inverse=False)
        logging.info(
            f"Model: MGNRP | "
            f"Motion: {motion:<12} | "
            f"Number of points: {N:<6} | "
            f"Chamfer: {chamfer_distance:.4f}"
        )
        pass
