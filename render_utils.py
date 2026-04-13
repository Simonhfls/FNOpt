import logging
import os
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from matplotlib import pyplot as plt, animation
from tqdm import tqdm
from tri_to_quad_mesh import batch_trimesh_to_quadmesh_torch
from utils import generate_ffmpeg_cmd, get_unique_filename


def setup_evaluation_logger(evaluate: bool,
                            model_name: str,
                            resolution: int,
                            postfix: str,
                            base_log_dir: str = "evaluation/S_MGNRP") -> None:
    """
    If `evaluate` is True:
      1. Create the log directory if needed.
      2. Configure the root logger to write both to file and to console.
      3. Silence matplotlib’s own log messages below WARNING.
    """
    if not evaluate:
        return

    # Construct the log file path
    log_filename = f"eval_log_{model_name}{resolution}_{postfix}.txt"
    log_path = Path(base_log_dir) / log_filename
    print(f"[INFO] Logging evaluation results to: {log_path.resolve()}")

    # Ensure parent directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging: file + console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler()
        ]
    )

    # Reduce verbosity of matplotlib logs
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def check_frame(pos: torch.Tensor,
                faces: torch.Tensor,
                frame_idx: int,
                max_abs: float = 1e8):
    """
    Raises an AssertionError if any of the following hold for this frame:
      • empty tensor
      • NaN or Inf
      • any entry > max_abs in absolute value
      • all zeros
      • face index out of range
    """
    # empty or zero-length
    if pos.numel() == 0 or pos.shape[0] == 0:
        raise AssertionError(f"[Frame {frame_idx}] predicted_pos is empty")

    # NaN or Inf anywhere
    if not torch.isfinite(pos).all():
        raise AssertionError(f"[Frame {frame_idx}] NaN or Inf detected in predicted_pos")

    # excessively large magnitudes
    if torch.any(pos.abs() > max_abs):
        raise AssertionError(f"[Frame {frame_idx}] entries exceed ±{max_abs}")

    # all zeros?
    if torch.all(pos == 0):
        raise AssertionError(f"[Frame {frame_idx}] all-zero predicted_pos")

    # valid face indices?
    max_face = int(faces.max())
    if max_face >= pos.shape[0]:
        raise AssertionError(
            f"[Frame {frame_idx}] invalid face index: max face {max_face} ≥ {pos.shape[0]}"
        )


def generate_heatmap_video(render_dir: str,
                           filename_template: str,
                           fps: int,
                           n_frames: int):
    """
    Runs ffmpeg to stitch all .pngs in render_dir into a single video,
    then cleans up the directory.
    """
    render_path = Path(render_dir)
    # make sure it exists
    if not render_path.is_dir():
        return

    # pick an output filename that doesn't clash
    output_file = get_unique_filename(filename_template,
                                      output_dir=str(render_path.parent))
    ffmpeg_cmd = generate_ffmpeg_cmd(
        render_dir=str(render_path),
        output_dir=str(render_path.parent),
        output_file=output_file,
        framerate=fps,
        n_frames=n_frames
    )

    print(f"Generating accuracy heatmap video: {output_file}")
    start = time.perf_counter()
    try:
        subprocess.run(ffmpeg_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error generating heatmap video: {e}")
    else:
        elapsed = time.perf_counter() - start
        print(f"Heatmap video done in {elapsed:.2f}s")

    # clean up .pngs & try to rmdir
    for img in render_path.glob("*.png"):
        img.unlink()
    try:
        render_path.rmdir()
        print(f"Deleted render directory: {render_path}")
    except OSError:
        print(f"Could not delete {render_path} (not empty?)")


def try_acquire_lock(path: str) -> bool:
    """
    Try to create a “.lock” file next to `path`.
    Returns True if we’ve created it successfully (lock acquired),
    False if it already exists.
    """
    lock_path = Path(path + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # use low-level os.open so we fail if it exists
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False

def release_lock(path: str) -> None:
    """
    Remove the corresponding .lock file.
    """
    try:
        os.remove(path + ".lock")
    except FileNotFoundError:
        pass


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

            # add xyz labels and title
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
    pbar = tqdm(total=num_steps)
    writervideo = animation.FFMpegWriter(fps=fps)
    anima.save(result_path, writer=writervideo,
               progress_callback=lambda i, n: pbar.update(1))


def render_video_with_gt(
    x_preds,             # tensor of shape (F, 3, H, W)
    bc_mask_list,            # list of np.array, each (H, W)
    save_path,
    gt_vertices_list=None,   # tensor of shape (F,V,3) or None
    gt_faces=None,           # tensor of shape (n_faces,3) or None
    gt_uv=None,
    dpi=400,
    azim=45,
    elev=45,
    zlim=None,
    xlim=None,
    ylim=None,
    fps=50,
    title_prefix="timestep",
    debug=False,
    gt_alpha=0.4,
    pred_alpha=1.0,
    show_handle_points=True,
    show_error_map=True
):
    F = len(x_preds)
    assert F == len(bc_mask_list), "Length of x_pred_list and bc_mask_list must match"

    has_gt = (
        isinstance(gt_vertices_list, torch.Tensor) and
        gt_vertices_list.ndim == 3
        # and gt_vertices_list.shape[0] == F
        and isinstance(gt_faces, torch.Tensor)
        and gt_faces.ndim == 2 and gt_faces.shape[1] == 3
    )

    print('has_gt:', has_gt)
    if has_gt:
        quad_verts, quad_quads = batch_trimesh_to_quadmesh_torch(
            gt_vertices_list, gt_faces, gt_uv,
            resolution=(x_preds.shape[-2], x_preds.shape[-1])
        )
        max_error = 0.5

        gt_faces = gt_faces.cpu().numpy()
        quad_verts = quad_verts.cpu().numpy()

    x_preds = x_preds.cpu().numpy()
    if xlim is None or ylim is None or zlim is None:
        if has_gt:
            all_x = quad_verts[...,0].ravel()
            all_y = quad_verts[...,1].ravel()
            all_z = quad_verts[...,2].ravel()
        else:
            all_x = np.concatenate([p[0].ravel() for p in x_preds])
            all_y = np.concatenate([p[1].ravel() for p in x_preds])
            all_z = np.concatenate([p[2].ravel() for p in x_preds])

        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()
        z_min, z_max = all_z.min(), all_z.max()
        x_center = 0.5 * (x_min + x_max)
        y_center = 0.5 * (y_min + y_max)
        z_center = 0.5 * (z_min + z_max)

        x_size = x_max - x_min
        y_size = y_max - y_min
        z_size = z_max - z_min

        max_range = max(x_size, y_size, z_size) * 0.55  # radius

        xlim = xlim or (x_center - max_range, x_center + max_range)
        ylim = ylim or (y_center - max_range, y_center + max_range)
        zlim = zlim or (z_center - max_range, z_center + max_range)

    fig = plt.figure(figsize=(4, 4), dpi=dpi)

    ax_pred = fig.add_subplot(111, projection='3d', zorder=1)
    ax_pred.patch.set_alpha(0)
    # ax_pred.set_axis_off()
    ax_pred.set_xlim(*xlim)
    ax_pred.set_ylim(*ylim)
    ax_pred.set_zlim(*zlim)
    ax_pred.view_init(elev=elev, azim=azim)
    ax_pred.set_box_aspect([1, 1, 1])

    if has_gt:
        ax_gt = fig.add_subplot(111, projection='3d', zorder=2)
        ax_gt.patch.set_alpha(0)
        ax_gt.set_axis_off()
        ax_gt.set_xlim(*xlim)
        ax_gt.set_ylim(*ylim)
        ax_gt.set_zlim(*zlim)
        ax_gt.view_init(elev=elev, azim=azim)
        ax_gt.set_box_aspect([1, 1, 1])

    def animate(i):
        ax_pred.cla()
        if has_gt:
            ax_gt.cla()

        p = x_preds[i]
        pred_pts = np.stack([p[0], p[1], p[2]], axis=-1)  # (H,W,3)
        # pred_pts = torch.stack([p[0], p[1], p[2]], dim=-1)


        if pred_alpha > 0:
            if show_error_map:
                # error-map coloring
                err_map = np.linalg.norm(pred_pts - quad_verts[i], axis=2)
                norm_err = np.clip(err_map / max_error, 0, 1)
                facecolors = plt.cm.jet(norm_err)
                ax_pred.plot_surface(
                    pred_pts[..., 0], pred_pts[..., 1], pred_pts[..., 2],
                    facecolors=facecolors, shade=False, alpha=pred_alpha,
                    rstride=1, cstride=1, linewidth=0, antialiased=False,
                    zorder=1
                )
            else:
                ax_pred.plot_surface(
                    pred_pts[..., 0], pred_pts[..., 1], pred_pts[..., 2],
                    color='C0',  # use matplotlib default color cycle, e.g. first color 'C0'
                    alpha=0.8,  # semi-transparent to see background wireframe
                    edgecolor='none',  # remove edge lines
                    antialiased=True,
                    linewidth=0
                )
            if show_handle_points:
                # print('show bc points')
                bc = bc_mask_list[i]
                cond = np.nonzero(bc)
                ax_pred.scatter(
                    pred_pts[cond[0], cond[1], 0],
                    pred_pts[cond[0], cond[1], 1],
                    pred_pts[cond[0], cond[1], 2],
                    marker='o', color='g', depthshade=False,
                    zorder=2
                )

        if has_gt and gt_alpha > 0:
            v = gt_vertices_list[i]

            ax_gt.plot_trisurf(
                v[:, 0], v[:, 1], v[:, 2],
                triangles=gt_faces,  # specify triangle indices
                color='lightgray',  # face fill color
                edgecolor='none',  # disable edge lines
                linewidth=0,  # zero line width
                antialiased=False,  # disable anti-aliasing
                alpha=gt_alpha,  # transparency
                zorder=1
            )

        for A in (ax_pred, (ax_gt if has_gt else ax_pred)):
            A.patch.set_alpha(0)
            A.set_axis_off()
            A.set_xlim(*xlim)
            A.set_ylim(*ylim)
            A.set_zlim(*zlim)
            A.view_init(elev=elev, azim=azim)

        ax_pred.set_title(f"{title_prefix}: {i}", pad=10)


        if debug and (i % 10 == 0):
            plt.draw()
            plt.pause(0.001)

        return fig,

    os.makedirs(Path(save_path).parent, exist_ok=True)
    pbar   = tqdm(total=F, desc="Rendering video")
    writer = animation.FFMpegWriter(fps=fps)
    anim   = animation.FuncAnimation(fig, animate, frames=F)
    anim.save(save_path, writer=writer,
              dpi=dpi,
              progress_callback=lambda i, n: pbar.update(1))
    plt.close(fig)