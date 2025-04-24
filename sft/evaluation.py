import pathlib
import torch
import pytorch3d.loss

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


def computeChamferDistance(ground_truth_point_clouds, our_point_clouds, min_index, max_index, point_clouds_lengths):
    # ground_truth_point_clouds = ground_truth_point_clouds.cpu()
    # our_point_clouds = our_point_clouds.cpu()
    # point_clouds_lengths = point_clouds_lengths.cpu()
    our_point_clouds[..., 0] = -our_point_clouds[..., 0]
    our_point_clouds[..., 1] = -our_point_clouds[..., 1]
    chamfer_distances = 1e4 * pytorch3d.loss.chamfer_distance(
                                  our_point_clouds[min_index:max_index],
                                  ground_truth_point_clouds[min_index:max_index],
                                  x_lengths=point_clouds_lengths[min_index:max_index],
                                  y_lengths=point_clouds_lengths[min_index:max_index],
                                  batch_reduction=None,
                              )[0]
    return torch.mean(chamfer_distances)

def computeMeshDistance(gt_mesh, our_mesh, min_index, max_index):

    # frame_3d_error = torch.norm(gt_mesh[min_index:max_index] - our_mesh[min_index:max_index]) / torch.norm(gt_mesh[min_index:max_index])
    per_vertex_errors = (gt_mesh[min_index:max_index] - our_mesh[min_index:max_index]).norm(dim=2)
    gt_mesh_norm = gt_mesh[min_index:max_index].norm(dim=2)
    normalized_error = torch.mean(per_vertex_errors.norm(dim=1) / gt_mesh_norm.norm(dim=1), 0)

    return normalized_error
