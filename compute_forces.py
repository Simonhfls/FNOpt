
import torch
from torch.nn.functional import normalize
import torch_scatter

import platform
if platform.machine() == 'AMD64' or platform.machine() == 'x86_64':
    device = torch.device('cuda')
elif platform.machine() == 'arm64':
    device = torch.device('cpu')

def compute_wind_force(cloth_m_npy, cloth_f_area_npy, face_tensor, cloth_v, cloth_pos, wind_density=1.0):
    """
    cloth_m_npy: (n_vertices, 1)
    cloth_f_area_npy: (n_faces, 1)
    face_tensor: (n_faces, 3) tensor of vertex indices
    cloth_v: (n_vertices, 3) tensor of vertex velocities
    cloth_pos: (n_vertices, 3) tensor of vertex positions

    return: (n_vertices, 3)
    """
    
    #some constants
    wind_v = torch.zeros(face_tensor.shape).to(device)
    #wind_v = torch.tensor([2.5,0.,3.]).repeat(face_tensor.shape[0], 1).to(device)
    wind_density = torch.ones(face_tensor.shape[0], 1).to(device) * wind_density
    wind_drag = torch.zeros(face_tensor.shape[0], 1).to(device)

    fext_list = torch.zeros_like(cloth_m_npy)

    vface = (torch.index_select(input=cloth_v, dim=0, index=face_tensor[:,0]) + 
                torch.index_select(input=cloth_v, dim=0, index=face_tensor[:,1]) +
                torch.index_select(input=cloth_v, dim=0, index=face_tensor[:,2]))/3
    
    vrel = wind_v - vface

    cross_p = torch.cross(torch.index_select(input=cloth_pos, dim=0, index=face_tensor[:,1]) -
                            torch.index_select(input=cloth_pos, dim=0, index=face_tensor[:,0]),
                            torch.index_select(input=cloth_pos, dim=0, index=face_tensor[:,2]) -
                            torch.index_select(input=cloth_pos, dim=0, index=face_tensor[:,0]))

    face_normal = normalize(cross_p, p=2.0)

    vn = (face_normal * vrel).sum(1)

    vt = vrel - vn.unsqueeze(1) * face_normal
    
    wind_force = wind_density * cloth_f_area_npy * torch.abs(vn).unsqueeze(1) * vn.unsqueeze(1) * face_normal + wind_drag * cloth_f_area_npy * vt

    scatter_0 = torch_scatter.scatter_add(wind_force/3, face_tensor[:,0].type(torch.int64), dim=0, dim_size=fext_list.shape[0])
    scatter_1 = torch_scatter.scatter_add(wind_force/3, face_tensor[:,1].type(torch.int64), dim=0, dim_size=fext_list.shape[0])
    scatter_2 = torch_scatter.scatter_add(wind_force/3, face_tensor[:,2].type(torch.int64), dim=0, dim_size=fext_list.shape[0])
    
    fext_list = fext_list + scatter_0 + scatter_1 + scatter_2
    
    return fext_list.float()
