import os
import pathlib
import numpy as np
import torch
import pytorch3d.loss
from generate_json_conf import get_path_from_gt_input
from tri_to_quad_mesh import load_obj_with_uv


def loadGroundTruth(scene_parameters, device='cpu'):
    pathlist = [str(p) for p in sorted(pathlib.Path(scene_parameters["ground_truth_dir"]).rglob("*.pt"))]

    max_n_points = 0
    for path in pathlist:
        temp = torch.load(path, weights_only=True).to(device)
        if temp.shape[0] > max_n_points:
            max_n_points = temp.shape[0]

    point_clouds = torch.zeros((len(pathlist), max_n_points, temp.shape[1]), device=device)
    point_clouds_lengths = torch.zeros(len(pathlist), dtype=torch.int64, device=device)
    counter = 0
    for path in pathlist:
        temp = torch.load(path, weights_only=True).to(device)
        point_clouds[counter, :temp.shape[0]] = temp
        point_clouds_lengths[counter] = temp.shape[0]
        counter += 1

    return point_clouds, point_clouds_lengths

def loadGroundTruthMGNRP(scene_parameters, input_data, device='cpu'):
    gt_path = get_path_from_gt_input(input_data, scene_parameters['root_mgnrp'])
    gt_x = torch.from_numpy(np.load(os.path.join(gt_path, "cloth_pos.npy"))).to(device)
    _, f, _ = load_obj_with_uv(scene_parameters['mesh_file_mgn'])
    f = torch.from_numpy(f).to(device)
    # sample points from the mesh
    n_points = 10000
    point_clouds = torch.zeros((gt_x.shape[0], n_points, 3), device=device)
    point_clouds_lengths = torch.tensor([n_points] * gt_x.shape[0], dtype=torch.int64, device=device)


    for i in range(gt_x.shape[0]):
        # sample points from the mesh
        point_clouds[i] = sampleMesh(n_points, gt_x[i], f, device=device)
    return point_clouds, point_clouds_lengths



def sampleMesh(number, vertices, triangles, device='cpu'):
    # print('sampling mesh device:', device)
    triangle_vertices = vertices[triangles]
    edges = triangle_vertices[:, 1:] - triangle_vertices[:, 0].unsqueeze(1)
    areas = torch.linalg.norm(torch.linalg.cross(edges[:, 0], edges[:, 1], dim = 1), dim = 1) # double of areas
    triangle_samples = torch.multinomial(input = areas, num_samples = number, replacement = True)

    uv = torch.rand([number, 2], device=device)
    u = 1 - torch.sqrt(uv[:, 0])
    v = torch.sqrt(uv[:, 0]) * (1 - uv[:, 1])
    w = uv[:, 1] * torch.sqrt(uv[:, 0])
    uvw = torch.stack([u, v, w], dim = 1).unsqueeze_(2)

    # print('triangle_vertices:', triangle_vertices.device)
    # print('triangle_samples:', triangle_samples.device)
    # print('uvw:', uvw.device)
    points = torch.sum(triangle_vertices[triangle_samples] * uvw, dim = 1)
    return points


def computeChamferDistance(ground_truth_point_clouds, our_point_clouds, min_index, max_index, point_clouds_lengths, inverse=True, frame_wise=False, scale=1e4):
    if inverse:
        our_point_clouds[..., 0] = -our_point_clouds[..., 0]
        our_point_clouds[..., 1] = -our_point_clouds[..., 1]
    chamfer_distances = scale * pytorch3d.loss.chamfer_distance(
                                  our_point_clouds[min_index:max_index],
                                  ground_truth_point_clouds[min_index:max_index],
                                  x_lengths=point_clouds_lengths[min_index:max_index],
                                  y_lengths=point_clouds_lengths[min_index:max_index],
                                  batch_reduction=None,
                              )[0]
    if frame_wise:
        return chamfer_distances
    return torch.mean(chamfer_distances)

def computeChamferDistanceFramewise(ground_truth_point_clouds, our_point_clouds, min_index, max_index, point_clouds_lengths, inverse=True, scale=1e4):
    if inverse:
        our_point_clouds[..., 0] = -our_point_clouds[..., 0]
        our_point_clouds[..., 1] = -our_point_clouds[..., 1]
    chamfer_distances = scale * pytorch3d.loss.chamfer_distance(
                                  our_point_clouds[min_index:max_index],
                                  ground_truth_point_clouds[min_index:max_index],
                                  x_lengths=point_clouds_lengths[min_index:max_index],
                                  y_lengths=point_clouds_lengths[min_index:max_index],
                                  batch_reduction=None,
                              )[0]
    return chamfer_distances

def computeMeshDistance(gt_mesh, our_mesh, min_index, max_index, frame_wise=False):

    # frame_3d_error = torch.norm(gt_mesh[min_index:max_index] - our_mesh[min_index:max_index]) / torch.norm(gt_mesh[min_index:max_index])
    per_vertex_errors = (gt_mesh[min_index:max_index] - our_mesh[min_index:max_index]).norm(dim=2)
    gt_mesh_norm = gt_mesh[min_index:max_index].norm(dim=2)
    e3d = per_vertex_errors.norm(dim=1) / gt_mesh_norm.norm(dim=1)
    if frame_wise:
        return e3d

    # normalized_error = torch.mean(per_vertex_errors.norm(dim=1) / gt_mesh_norm.norm(dim=1), 0)
    return torch.mean(e3d, 0)
    # return normalized_error

def computeMeshDistanceWithoutNormalization(gt_mesh, our_mesh, min_index, max_index):
    # frame_3d_error = torch.norm(gt_mesh[min_index:max_index] - our_mesh[min_index:max_index]) / torch.norm(gt_mesh[min_index:max_index])
    per_vertex_errors = (gt_mesh[min_index:max_index] - our_mesh[min_index:max_index]).norm(dim=2)
    normalized_error = torch.mean(per_vertex_errors)

    return normalized_error


def savePointCloud(point_clouds, filename):
    # save to obj file
    with open(filename, 'w') as f:
        for i in range(point_clouds.shape[0]):
            f.write(f"v {point_clouds[i, 0]} {point_clouds[i, 1]} {point_clouds[i, 2]}\n")
            f.write("\n")  # separate point clouds by a blank line