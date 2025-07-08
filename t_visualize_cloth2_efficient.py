import shutil
import subprocess
import uuid

import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from dataset_cloth3 import DatasetCloth
from dataset_utils import DatasetToSingleChannel
from metamizer import get_Net3 as get_Net
from Logger import Logger
import torch
from get_param2 import params,toCuda,get_hyperparam, device
import time
import os
from pytorch3d.io import save_obj
from utils import grid_to_trimesh_faces, generate_ffmpeg_cmd

if __name__ == '__main__':
	print('params:', params)
	params.wandb.log = False
	params.training = False
	logger = Logger(get_hyperparam(params),use_csv=False,use_tensorboard=False)
	save = True#True#
	unique_id = uuid.uuid4().hex[:8]
	Y = params.inference.material.stretching
	S = params.inference.material.shearing
	B = params.inference.material.bending
	if save:
		path = (f"plots/{get_hyperparam(params).replace(' ','_').replace(';','_')}"
				f"/cloth/{params.inference.load_date_time}"
				f"/stiff_{Y} "
				f"shear_{S} "
				f"bend_{B} "
				f"iters_{params.inference.iterations_per_timestep}"
				f"/tmp{unique_id}")
		os.makedirs(path, exist_ok=True)
		assert os.path.exists(path), f"Failed to create path: {path}"
		print(f"Render directory created: {path}")

	frame = 0
	dpi = 200

	# metamizer = toDType(toCuda(get_Net(params)))
	metamizer = toCuda(get_Net(params))
	#metamizer.nn = torch.compile(metamizer.nn)
	if params.net.name not in ["MeshGraphNets2"]:
		date_time,index = logger.load_state(metamizer,None, datetime=params.inference.load_date_time,index=params.inference.load_index, device=device)
	else:
		date_time, index = logger.load_state_mgn2(metamizer,None, datetime=params.inference.load_date_time,index=params.inference.load_index, device=device)
	print(f"loaded: {date_time}, {index}")
	metamizer.eval()

	scales = []
	max_scales = []
	gradients = []
	n_frames = params.inference.rollout.n_frames
	# (1, 3, h, w)
	length_conversion = max(params.inference.height, params.inference.width)
	time_conversion = params.inference.rollout.framerate
	g_vect = toCuda(torch.tensor([0, 0, -9.81])).unsqueeze(0).unsqueeze(2).unsqueeze(3)  # gravity vector. CODO: radnom directions / strengths of gravity
	a_exts = toCuda(torch.ones(1, 3, params.inference.height,params.inference.width)) * g_vect
	a_exts = a_exts * length_conversion / 50**2  # convert to m/s^2
	with torch.no_grad():
		for epoch in range(1):
			original_dataset = DatasetCloth(params.inference.height,params.inference.width,1,1,n_frames,iterations_per_timestep=params.inference.iterations_per_timestep,stiffness_range=params.cloth.stretching_range,shearing_range=params.cloth.shearing_range,bending_range=params.cloth.bending_range,a_ext_range=params.cloth.g)
			original_dataset.reset0_env(0)	# bc is defined inside
			original_dataset.set_external_forces(a_exts)
			# print('rot.sum():', original_vdataset.rot_speed.sum())
			original_dataset.stiffnesses[:] = Y
			original_dataset.shearings[:] = S
			original_dataset.bendings[:] = B

			dataset = DatasetToSingleChannel(original_dataset)
			# dataset.ask_sft = torch.compile(dataset.ask_sft, backend="inductor", mode="reduce-overhead")
			print('height:', params.inference.height)
			print('width:', params.inference.width)
			print('iteration per timestep:',params.inference.iterations_per_timestep)
			print('total iterations:',n_frames*params.inference.iterations_per_timestep)

			FPS=0
			start_time = time.time()
			for t in range(n_frames*params.inference.iterations_per_timestep):
				if t % 50 == 0:
					print(f"t: {t}")
				# grads, hidden_states = dataset.ask()
				grads, hidden_states = dataset.ask_sft()
				update_steps, new_hidden_states = metamizer(grads, hidden_states)
				loss = dataset.tell(update_steps, new_hidden_states)
				scales.append(new_hidden_states[0][2][0,0,0,0].detach().cpu().numpy())
				gradients.append(torch.norm(grads, p=2).detach().cpu().numpy())

				if (t+1)%params.inference.iterations_per_timestep==0: # visualize only at a new timestep (a timestep can take several iterations to optimize)

					if params.inference.visualize_3d: # visualize 3D cloth
						index = 0

						x = original_dataset.x[index].cpu().numpy()
						a = original_dataset.a[index].cpu()
						bc_masks = original_dataset.bc_masks[index,0].cpu()

						ls = LightSource(azdeg=315, altdeg=45)  # Control the direction of the light
						rgb = ls.shade(x[2], cmap=plt.cm.viridis, vert_exag=0.1, blend_mode='soft')

						plt.figure(1,figsize=(800/dpi,800/dpi),dpi=dpi)
						plt.clf()
						fig, ax = plt.subplots(1,1,subplot_kw={"projection": "3d"},num=1,computed_zorder=False)

						surf = ax.plot_surface(x[0], x[1], x[2], linewidth=0.1, antialiased=False,zorder=4,rstride=1,cstride=1)#,alpha=0.5) # cloth surface

						# boundary conditions
						cond = (bc_masks > 0).nonzero()
						ax.scatter(x[0,cond[:,0],cond[:,1]],x[1,cond[:,0],cond[:,1]],x[2,cond[:,0],cond[:,1]],marker='o',color='g',depthshade=False,zorder=5) # boundarys conditions

						ax.grid(False)
						ax.set_axis_off()  # Completely removes the 3D box


						# Remove the axes (pane color)
						ax.xaxis.pane.fill = False
						ax.yaxis.pane.fill = False
						ax.zaxis.pane.fill = False

						# Hide the axes lines
						ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # X-axis
						ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Y-axis
						ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Z-axis

						# Optionally, hide the ticks
						ax.set_xticks([])
						ax.set_yticks([])
						ax.set_zticks([])


						"""
						q_stride, q_l=8,10
						# gradients
						ax.quiver(x[0,::q_stride,::q_stride], x[1,::q_stride,::q_stride], x[2,::q_stride,::q_stride], \
							q_l*grads[0,0,::q_stride,::q_stride], q_l*grads[0,1,::q_stride,::q_stride], q_l*grads[0,2,::q_stride,::q_stride],color='r')

						# accelerations
						ax.quiver(x[0,::q_stride,::q_stride], x[1,::q_stride,::q_stride], x[2,::q_stride,::q_stride], \
							q_l*a[0,::q_stride,::q_stride], q_l*a[1,::q_stride,::q_stride], q_l*a[2,::q_stride,::q_stride],color='g')
						"""


						ax.set_zlim(-2*params.inference.height*0.6, 1.01)
						ax.set_xlim(-params.inference.height*0.6, params.inference.height*0.6)
						ax.set_ylim(-params.inference.height*0.6, params.inference.height*0.6)
						plt.title(f"timestep: {original_dataset.T[index].cpu().numpy()[0]}")

						if save:
							plt.savefig(f"{path}/frame_{str(frame).zfill(4)}.png",dpi=dpi)
							frame += 1

						plt.draw()
						plt.pause(0.01)

						if params.inference.save_obj:
							# save mesh
							mesh_path = f"{path}/frame_{str(frame).zfill(4)}.obj"
							print('mesh_path:', mesh_path)
							save_obj(mesh_path, torch.from_numpy(x).permute(1, 2, 0).reshape(-1, 3),torch.from_numpy(grid_to_trimesh_faces(num_rows=params.inference.height, num_cols=params.inference.width)))

					if params.inference.visualize_scaling: # visualize, how scaling changes during update steps
						plt.figure(2)
						plt.clf()
						stride = 1#len(scales)//200+1
						plt.semilogy(scales[::stride])
						plt.xlabel("iteration")
						plt.ylabel("scale")
						plt.legend(["scales"])
						plt.draw()
						plt.pause(0.01)

					if params.inference.visualize_grads: # visualize, how norm of loss gradients changes during update steps
						plt.figure(3,figsize=(1600/dpi,800/dpi),dpi=dpi)
						plt.clf()
						stride = 1#len(scales)//200+1
						plt.semilogy(gradients[::stride])
						plt.xlabel("iteration")
						plt.ylabel("gradient norm")
						plt.title(f"Gradient Norm reduction of Metamizer, {params.inference.iterations_per_timestep} iterations per timestep, {params.inference.height} x {params.inference.width}")
						plt.draw()
						plt.pause(0.01)

			end_time = time.time()
			print(f"dt = {end_time-start_time}s")
			print(f"FPS: {n_frames/(end_time-start_time)}")


	if params.inference.visualize_scaling:  # visualize, how scaling changes during update steps
		plt.figure(2)
		plt.clf()
		stride = 1  # len(scales)//200+1
		plt.semilogy(scales[::stride])
		plt.xlabel("iteration")
		plt.ylabel("scale")
		plt.legend(["scales"])
		plt.draw()
		plt.savefig(f"{path}/scale_{str(frame).zfill(4)}.png", dpi=dpi)

	if params.inference.visualize_grads:  # visualize, how norm of loss gradients changes during update steps
		plt.figure(3, figsize=(1600 / dpi, 800 / dpi), dpi=dpi)
		plt.clf()
		stride = 1  # len(scales)//200+1
		plt.semilogy(gradients[::stride])
		plt.xlabel("iteration")
		plt.ylabel("gradient norm")
		plt.title(
			f"Gradient Norm reduction of Metamizer, {params.inference.iterations_per_timestep} iterations per timestep, {params.inference.height} x {params.inference.width}")
		plt.draw()
		plt.savefig(f"{path}/grad_norm_{str(frame).zfill(4)}.png", dpi=dpi)

	output_file = f"V_iterpstep{params.inference.iterations_per_timestep}_res{params.inference.height}x{params.inference.width}_Y{Y}_S{S}_B{B}.mp4"

	ffmpeg_cmd = generate_ffmpeg_cmd(
		render_dir=path,
		output_file=output_file,
		output_dir=os.path.dirname(path),
		framerate=params.inference.rollout.framerate,
		n_frames=params.inference.rollout.n_frames
	)
	# execute ffmpeg to render images
	try:
		start_time = time.perf_counter()
		subprocess.run(ffmpeg_cmd, check=True)
		end_time = time.perf_counter()
		print(f"Render video generation completed in {(end_time - start_time):.2f} seconds")
		print('Video saved to:', os.path.abspath(os.path.join(os.path.dirname(path), output_file)))
	except subprocess.CalledProcessError as e:
		print(f"Error generating video: {e}")
	else:
		# delete all frame png files in render directory
		for file in os.listdir(path):
			if file.endswith(".png") and file.startswith('frame'):
				os.remove(os.path.join(path, file))

	try:
		os.rmdir(path)  # Only works if the directory is empty
		print(f"Deleted render directory: {path}")
	except OSError:
		# Fallback: force delete the whole folder (if any residual files remain)
		shutil.rmtree(path)
		print(f"Force-deleted render directory with residual files: {path}")

