import os
from pathlib import Path

import numpy as np
import torch

from matplotlib import pyplot as plt, animation
from matplotlib.colors import LightSource
from pytorch3d.renderer import (TexturesUV, BlendParams,
                                MeshRasterizer, RasterizationSettings,
                                HardPhongShader, PointLights)
from pytorch3d.structures import Meshes
import tqdm


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

    projection = torch.zeros([4, 4], dtype=torch.float32)
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


def render_single0(position_list, face_info, viewport, result_path, fps, debug=False):
    plot_size = len(position_list)  # plus 1 for the gt
    fig, axs = plt.subplots(1, plot_size, subplot_kw={'projection': '3d'})
    if len(position_list) == 1:
        axs = [axs]
    fig.set_size_inches(19.2, 10.8)

    azim = viewport[0]
    elev = viewport[1]

    min_length = 99999999999
    for single_plot_list in position_list:
        for position in single_plot_list:
            if position.shape[0] < min_length:
                num_steps = position.shape[0]

    # num_steps = 120

    # compute bounds
    all_bounds_min = []
    all_bounds_max = []
    for single_plot_list in position_list:
        for plot in single_plot_list:
            plot = plot[:num_steps]
            bb_min = np.squeeze(plot).min(axis=(0, 1))
            bb_max = np.squeeze(plot).max(axis=(0, 1))
            all_bounds_min.append(bb_min)
            all_bounds_max.append(bb_max)
    final_bound_min = np.stack(all_bounds_min).min(axis=0)
    final_bound_max = np.stack(all_bounds_max).max(axis=0)
    # get the max range
    ran_val = (final_bound_max - final_bound_min).max()
    # get the mean
    mean_val = (final_bound_max + final_bound_min) / 2
    bound = (mean_val - ran_val / 2, mean_val + ran_val / 2)

    # bound = (final_bound_min, final_bound_max)

    def animate(num):
        # print(num)
        for plot_group_index, plot_group in enumerate(position_list):
            axs[plot_group_index].cla()
            axs[plot_group_index].set_xlim([bound[0][0], bound[1][0]])
            axs[plot_group_index].set_ylim([bound[0][1], bound[1][1]])
            axs[plot_group_index].set_zlim([bound[0][2], bound[1][2]])

            axs[plot_group_index].azim = azim
            axs[plot_group_index].elev = elev

            # ✅ add xyz labels and title
            axs[plot_group_index].set_xlabel("X")
            axs[plot_group_index].set_ylabel("Y")
            axs[plot_group_index].set_zlabel("Z")

            for plot_index, position in enumerate(plot_group):
                pos = position[num]
                # pos = transform_coordinates(position[num])

                if plot_index == 0:
                    alpha = 1
                else:
                    alpha = 0.3
                # if ind < 7:
                #     alpha = 1
                #     color = 'blue'
                # else:
                #     alpha = 0.5
                #     color = 'red'

                axs[plot_group_index].plot_trisurf(pos[:, 0], pos[:, 1], face_info, pos[:, 2], shade=True, alpha=alpha)


        fig.suptitle("azim %d | elev %d | frame %d" % (azim, elev, num))

        if debug:
            if num % 10 == 0:
                plt.draw()
                plt.pause(0.001)

        return fig,

    anima = animation.FuncAnimation(fig, animate, frames=num_steps)
    pbar = tqdm.tqdm(total=num_steps)
    writervideo = animation.FFMpegWriter(fps=fps)
    anima.save(result_path, writer=writervideo,
               progress_callback=lambda i, n: pbar.update(1))

def render_single(position_list, face_info_list, viewport, result_path, fps, debug=False):
    plot_size = len(position_list)  # plus 1 for the gt
    fig, axs = plt.subplots(1, plot_size, subplot_kw={'projection': '3d'})
    if len(position_list) == 1:
        axs = [axs]
    fig.set_size_inches(19.2, 10.8)

    azim = viewport[0]
    elev = viewport[1]

    min_length = 99999999999
    for single_plot_list in position_list:
        for position in single_plot_list:
            if position.shape[0] < min_length:
                num_steps = position.shape[0]

    # num_steps = 120
    bound_thresh = 2.0


    # compute bounds
    all_bounds_min = []
    all_bounds_max = []
    for single_plot_list in position_list:
        for plot in single_plot_list:
            plot = plot[:num_steps]
            plot = np.clip(plot, -bound_thresh, bound_thresh)   # for clip diverged values

            bb_min = np.squeeze(plot).min(axis=(0, 1))
            bb_max = np.squeeze(plot).max(axis=(0, 1))
            all_bounds_min.append(bb_min)
            all_bounds_max.append(bb_max)
    final_bound_min = np.stack(all_bounds_min).min(axis=0)
    final_bound_max = np.stack(all_bounds_max).max(axis=0)
    # get the max range
    ran_val = (final_bound_max - final_bound_min).max()
    # get the mean
    mean_val = (final_bound_max + final_bound_min) / 2
    bound = (mean_val - ran_val / 2, mean_val + ran_val / 2)

    # bound = (final_bound_min, final_bound_max)

    def animate(num):
        # print(num)
        for plot_group_index, plot_group in enumerate(position_list):
            axs[plot_group_index].cla()
            axs[plot_group_index].set_xlim([bound[0][0], bound[1][0]])
            axs[plot_group_index].set_ylim([bound[0][1], bound[1][1]])
            axs[plot_group_index].set_zlim([bound[0][2], bound[1][2]])

            axs[plot_group_index].azim = azim
            axs[plot_group_index].elev = elev

            # ✅ add xyz labels and title
            axs[plot_group_index].set_xlabel("X")
            axs[plot_group_index].set_ylabel("Y")
            axs[plot_group_index].set_zlabel("Z")

            for plot_index, position in enumerate(plot_group):
                pos = position[num]
                face_info = face_info_list[plot_group_index][plot_index]
                # pos = transform_coordinates(position[num])

                if plot_index == 0:
                    alpha = 1
                else:
                    alpha = 0.3
                # if ind < 7:
                #     alpha = 1
                #     color = 'blue'
                # else:
                #     alpha = 0.5
                #     color = 'red'

                axs[plot_group_index].plot_trisurf(pos[:, 0], pos[:, 1], face_info, pos[:, 2], shade=True, alpha=alpha)


        fig.suptitle("azim %d | elev %d | frame %d" % (azim, elev, num))

        if debug:
            if num % 10 == 0:
                plt.draw()
                plt.pause(0.001)

        return fig,

    anima = animation.FuncAnimation(fig, animate, frames=num_steps)
    pbar = tqdm.tqdm(total=num_steps)
    writervideo = animation.FFMpegWriter(fps=fps)
    anima.save(result_path, writer=writervideo,
               progress_callback=lambda i, n: pbar.update(1))


def render_metamizer_video(
    x_list,                         # list of np.array, shape = (3, H, W)
    bc_mask_list,                  # list of np.array, shape = (H, W)
    save_path,                     # path to save the video (e.g., output.mp4)
    dpi=200,
    azim=45,
    elev=45,
    figsize=(800, 800),
    zlim=None,
    xlim=None,
    ylim=None,
    fps=50,
    title_prefix="timestep",
    debug=False
):
    assert len(x_list) == len(bc_mask_list), "x_list and bc_mask_list must be the same length"
    num_frames = len(x_list)

    # Automatically compute axis limits if not provided
    if xlim is None or ylim is None or zlim is None:
        all_x = np.concatenate([x[0].flatten() for x in x_list])
        all_y = np.concatenate([x[1].flatten() for x in x_list])
        all_z = np.concatenate([x[2].flatten() for x in x_list])
        # margin = 4
        #
        # if xlim is None:
        #     xlim = (all_x.min() - margin, all_x.max() + margin)
        # if ylim is None:
        #     ylim = (all_y.min() - margin, all_y.max() + margin)
        # if zlim is None:
        #     zlim = (all_z.min() - margin, all_z.max() + margin)
        def get_bound(arr, ratio=0.1):  # ratio = margin 相对于范围
            min_val = np.min(arr)
            max_val = np.max(arr)
            margin = (max_val - min_val) * ratio
            return (min_val - margin, max_val + margin)

        xlim = xlim or get_bound(all_x)
        ylim = ylim or get_bound(all_y)
        zlim = zlim or get_bound(all_z)

    fig = plt.figure(figsize=(figsize[0] / dpi, figsize[1] / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    def animate(i):
        ax.clear()
        x_np = x_list[i]
        bc_mask = bc_mask_list[i]

        ls = LightSource(azdeg=315, altdeg=45)
        rgb = ls.shade(x_np[2], cmap=plt.cm.viridis, vert_exag=0.1, blend_mode='soft')
        # plt.figure(1, figsize=(800 / dpi, 800 / dpi), dpi=dpi)
        # plt.clf()
        surf = ax.plot_surface(
            x_np[0], x_np[1], x_np[2],
            linewidth=0.1, antialiased=False, zorder=4,
            rstride=1, cstride=1
        )

        cond = (bc_mask > 0).nonzero()
        ax.scatter(
            x_np[0, cond[0], cond[1]],
            x_np[1, cond[0], cond[1]],
            x_np[2, cond[0], cond[1]],
            marker='o', color='g', depthshade=False, zorder=5
        )

        ax.grid(False)
        ax.set_axis_off()

        # Remove pane fill and axis lines
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.fill = False
            axis.line.set_color((1.0, 1.0, 1.0, 0.0))
            axis.set_ticks([])

        ax.set_zlim(*zlim)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        ax.azim = azim
        ax.elev = elev
        ax.set_title(f"{title_prefix}: {i}")

        if debug:
            if i % 10 == 0:
                plt.draw()
                plt.pause(0.001)

        return fig,

    pbar = tqdm.tqdm(total=num_frames, desc="Rendering video")
    writer = animation.FFMpegWriter(fps=fps)

    os.makedirs(Path(save_path).parent, exist_ok=True)

    anim = animation.FuncAnimation(fig, animate, frames=num_frames)
    anim.save(save_path, writer=writer, progress_callback=lambda i, n: pbar.update(1))
    plt.close()