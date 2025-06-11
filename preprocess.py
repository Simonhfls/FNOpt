import torch
from pytorch3d.io import load_obj
from pytorch3d.transforms import Transform3d, Rotate, Translate, euler_angles_to_matrix
from pytorch3d.io import save_obj


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


if __name__ == '__main__':
    device ='cpu'
    # test this function
    ref_pos = torch.randn(10, 1024, 3)
    handle_ind = (0, 19)
    rot_mat, trans_mat = make_rot_mat_trans_mat(ref_pos, handle_ind)


    obj_path = '/Users/ruochen/Documents/liris_code/datasets/phi_sft/synthetic/template_untextured.obj'
    # pytorch3d load obj file
    verts, faces, aux = load_obj(obj_path)
    verts = verts.unsqueeze(0)  # add batch dimension
    handle_ind = (3, 8)

    # apply a rotation and a translation to the vertices

    # 构建旋转矩阵（欧拉角 z=45°，绕 z 轴旋转）
    angles = torch.tensor([[25, 10, -20.0]], device=device)  # [B, 3], in degrees
    angles_rad = torch.deg2rad(angles)
    R = euler_angles_to_matrix(angles_rad, convention="XYZ")  # [B, 3, 3]

    # 构建平移向量
    T = torch.tensor([[0.1, 0.2, 0.3]], device=device)  # [B, 3]

    # 创建 Transform3d 实例并应用变换
    transform = Transform3d(device=device).rotate(R=R).translate(T)
    verts = transform.transform_points(verts)  # [B, V, 3]

    # save transformed verts to obj file
    save_obj('./transformed1.obj', verts[0], faces.verts_idx)


    rot_mat, trans_mat = make_rot_mat_trans_mat(verts, handle_ind)
    trans_vert = transform_positions(verts.permute(0, 2, 1), rot_mat, trans_mat)

    # save transformed verts to obj file
    save_obj('./transformed2.obj', trans_vert[0], faces.verts_idx)
