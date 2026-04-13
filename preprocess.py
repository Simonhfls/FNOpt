import torch

def make_rot_mat_trans_mat(ref_pos, handle_ind, global_to_local=False, device='cpu'):
    # positions shape: [frames x 1024 x 3]

    a = ref_pos[:, handle_ind[0], :] - ref_pos[:, handle_ind[1], :]
    a = a / torch.norm(a, dim=1, keepdim=True)
    # shape of a: [frames x 3]

    j = torch.tensor([0.0, 1.0, 0.0]).repeat(ref_pos.shape[0], 1).to(device)

    i = torch.cross(j, a, dim=1)

    k = torch.cross(i, j, dim=1)

    o = ref_pos[:, handle_ind[0], :]

    end_row = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(ref_pos.shape[0], 1, 1).to(device)

    aff_mat = torch.stack((i, j, k, o), 1).transpose(1, 2)
    aff_mat = torch.cat((aff_mat, end_row), 1)
    # shape of aff_mat: [frames x 4 x 4]

    if global_to_local == True:
        glob2loc_mat = aff_mat

    aff_mat = torch.inverse(aff_mat)

    if global_to_local == False:

        return aff_mat[:, 0:3, 0:3], aff_mat[:, 0:3, 3].reshape(ref_pos.shape[0], 1, 3)

    else:

        return aff_mat[:, 0:3, 0:3], aff_mat[:, 0:3, 3].reshape(ref_pos.shape[0], 1, 3), glob2loc_mat[:, 0:3,
                                                                                         0:3], glob2loc_mat[:, 0:3,
                                                                                               3].reshape(
            ref_pos.shape[0], 1, 3)


def transform_positions(positions, rot_mat, trans_mat):

    #positions shape : [frames x 3 x 1024]
    #affine matrix shape : [frames x 3 x 3]
    #translate shape: [frames x 1 x 3]

    rot_pos = torch.bmm(rot_mat, positions).transpose(1,2)

    trans_pos = rot_pos + trans_mat

    return trans_pos


def generate_correction_traj(target, source, mask, n_steps=20, device=None):
    """
    Generates a smooth trajectory from rest_positions to target initial handle_traj[0]

    Args:
        target (torch.Tensor): Target position of shape (3, H, W)
        source (torch.Tensor): Rest position of shape (N, 3)
        mask (torch.Tensor): Handle mask of shape (3, H, W) (True for non-handle)
        n_steps (int): Number of interpolation steps
        device (torch.device): Target device

    Returns:
        torch.Tensor: Interpolated trajectory of shape (n_steps, 3, H, W)
    """
    if device is None:
        device = target.device

    B, H, W = target.shape
    rest = source.clone().reshape(H, W, 3).permute(2, 0, 1)  # (3, H, W)
    target = target
    mask = mask.reshape(H, W, 3).permute(2, 0, 1)  # (3, H, W)
    rest[mask] = 0
    correction_traj = []

    for i in range(n_steps):
        alpha = 0.5 * (1 - torch.cos(torch.tensor(i / (n_steps - 1) * torch.pi, device=device)))  # cosine easing
        interpolated = rest * mask + ((1 - alpha) * rest + alpha * target) * (~mask)
        correction_traj.append(interpolated.unsqueeze(0))  # (1, 3, H, W)
    return torch.cat(correction_traj, dim=0)  # (n_steps, 3, H, W)



