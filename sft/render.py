import numpy as np
import torch
from pytorch3d.renderer import (TexturesUV, BlendParams,
                                MeshRasterizer, RasterizationSettings,
                                HardPhongShader, PointLights)
from pytorch3d.structures import Meshes

def opencv_projection(
    image_size,     # width, height
    optical_center, # c in pixel
    focal_lengths,  # f in pixel
    z_near,
    z_far,
):
    optical_shifts = (image_size - 2 * optical_center) / image_size
    optical_shifts[0] *= -1
    relative_focal_lengths = -2 * focal_lengths / image_size

    projection = torch.zeros([4, 4], dtype=torch.float32, device="cuda")
    projection[0, 0] = relative_focal_lengths[0]
    projection[1, 1] = relative_focal_lengths[1]
    projection[0, 2] = optical_shifts[0]
    projection[1, 2] = optical_shifts[1]
    projection[2, 2] = -(z_far + z_near) / (z_far - z_near)
    projection[2, 3] = -2 * z_far * z_near / (z_far - z_near)
    projection[3, 2] = -1.0

    return projection


def render_pytorch(vertices, faces, texture, uvs, faces_uvs, cameras, image_size):
    device = vertices.device
    if len(vertices.shape) == 2:
        vertices = vertices.unsqueeze(0)
    if len(faces.shape) == 2:
        faces = faces.unsqueeze(0)

    tex = TexturesUV(verts_uvs=uvs.unsqueeze(0), faces_uvs=faces_uvs.unsqueeze(0), maps=texture)
    meshes = Meshes(verts=vertices, faces=faces, textures=tex.extend(vertices.shape[0]))

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


class MeshRendererWithDepth(torch.nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf

def render_nvdiffrast(
        context,
        vertices,
        triangles,
        vertex_attributes,
        texture,
        camera_transforms,
        resolution,
        antialias=True,
        crop=True,
        real_scene=False,
):
    import nvdiffrast.torch as dr

    device = vertices.device
    n = camera_transforms.shape[0]
    v = vertices.shape[0]

    vertices_hom = torch.cat([vertices, torch.ones([v, 1], device=device, dtype=torch.float32)], axis=1)
    vertices_pixel = torch.matmul(vertices_hom.expand(n, v, -1), torch.transpose(camera_transforms, -2, -1))

    rast, diff_rast = dr.rasterize(context, vertices_pixel, triangles, resolution=resolution)
    image_attributes, _ = dr.interpolate(vertex_attributes, rast, triangles, rast_db=diff_rast, diff_attrs=None)
    color = dr.texture(texture, uv=image_attributes[..., [0, 1]], filter_mode="linear")
    if antialias:
        color = dr.antialias(color, rast, vertices_pixel, triangles)
    images = torch.cat([color, torch.ones(*color.shape[:-1], 1, device=device)], dim=-1)
    if crop:
        if real_scene:
            images = torch.where(rast[..., 3:] > 0, images, torch.tensor([0.0], device=device))
        else:
            images = torch.where(rast[..., 3:] > 0, images, torch.tensor([1.0], device=device))     # synthetic sequence
        images[..., 3:] = torch.where(rast[..., 3:] > 0, images[..., 3:], torch.tensor([0.0], device=device))
    return images




def ComputeViewMatrix(camera_position, camera_forward, camera_up):
    right = np.cross(camera_forward, camera_up)
    right /= np.linalg.norm(right)

    direction_matrix = np.array([[right[0], camera_up[0], - camera_forward[0], 0.0],
                                 [right[1], camera_up[1], - camera_forward[1], 0.0],
                                 [right[2], camera_up[2], - camera_forward[2], 0.0],
                                 [0.0, 0.0, 0.0, 1.0]],
                                dtype=np.float32)
    position_matrix = np.array([[1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [- camera_position[0], - camera_position[1], - camera_position[2], 1.0]],
                               dtype=np.float32)

    return position_matrix @ direction_matrix