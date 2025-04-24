import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from dataset_cloth import DatasetCloth
#from setups_multistep_1_channel import Dataset
from dataset_utils import DatasetToSingleChannel
from metamizer import get_Net
from Logger import Logger
import torch
import numpy as np
from get_param import params,toCuda,toCpu,get_hyperparam,toDType
import time
import os

logger = Logger(get_hyperparam(params),use_csv=False,use_tensorboard=False)


save = True#True#
if save:
	path = f"plots/{get_hyperparam(params).replace(' ','_').replace(';','_')}/cloth/{params.inference.load_date_time.replace(';','_')}/stiff_{params.stiffness} shear_{params.shearing} bend_{params.bending}"
	os.makedirs(path,exist_ok=True)
frame = 0
dpi = 200

metamizer = toDType(toCuda(get_Net(params)))
#metamizer.nn = torch.compile(metamizer.nn)
device = 'cuda' if params.cuda else 'cpu'

date_time,index = logger.load_state(metamizer,None,datetime=params.load_date_time,index=params.load_index, device=device)
print(f"loaded: {date_time}, {index}")
metamizer.eval()

scales = []
max_scales = []
gradients = []

with torch.no_grad():
	for epoch in range(1):
		original_dataset = DatasetCloth(params.height,params.width,1,1,params.average_sequence_length,iterations_per_timestep=params.iterations_per_timestep,stiffness_range=params.stiffness_range,shearing_range=params.shearing_range,bending_range=params.bending_range,a_ext_range=params.g)
		original_dataset.reset0_env(0)	# bc is defined inside
		
		original_dataset.stiffnesses[:] = params.stiffness
		original_dataset.shearings[:] = params.shearing
		original_dataset.bendings[:] = params.bending
		
		dataset = DatasetToSingleChannel(original_dataset)
		FPS=0
		start_time = time.time()

		average_sequence_length = 5000
		for t in range(average_sequence_length*params.iterations_per_timestep):
			print(f"t: {t}")
			grads, hidden_states = dataset.ask()
			update_steps, new_hidden_states = metamizer(grads, hidden_states)
			loss = dataset.tell(update_steps, new_hidden_states)
			
			scales.append(new_hidden_states[0][2][0,0,0,0].detach().cpu().numpy())
			gradients.append(torch.norm(grads,p=2).detach().cpu().numpy())
			
			if (t+1)%params.iterations_per_timestep==0: # visualize only at a new timestep (a timestep can take several iterations to optimize)
				
				if True: # visualize 3D cloth
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
					

					ax.set_zlim(-2*params.height*0.6, 1.01)
					ax.set_xlim(-params.height*0.6, params.height*0.6)
					ax.set_ylim(-params.height*0.6, params.height*0.6)
					plt.title(f"timestep: {original_dataset.T[index].cpu().numpy()[0]}")
				
					if save:
						plt.savefig(f"{path}/{str(frame).zfill(4)}.png",dpi=dpi)
						frame += 1
					
					plt.draw()
					plt.pause(0.01)
				
				if False: # visualize, how scaling changes during update steps
					plt.figure(2)
					plt.clf()
					stride = 1#len(scales)//200+1
					plt.semilogy(scales[::stride])
					plt.xlabel("iteration")
					plt.ylabel("scale")
					plt.legend(["scales"])
					plt.draw()
					plt.pause(0.01)
				
				if False: # visualize, how norm of loss gradients changes during update steps
					plt.figure(3,figsize=(1600/dpi,800/dpi),dpi=dpi)
					plt.clf()
					stride = 1#len(scales)//200+1
					plt.semilogy(gradients[::stride])
					plt.xlabel("iteration")
					plt.ylabel("gradient norm")
					plt.title(f"Gradient Norm reduction of Metamizer, {params.iterations_per_timestep} iterations per timestep, {params.height} x {params.width}")
					
					if frame%3==0 and save:
						plt.savefig(f"{path}/grad_norm_{str(frame).zfill(4)}.png",dpi=dpi)
					plt.draw()
					plt.pause(0.01)
				
		
		end_time = time.time()
		print(f"dt = {end_time-start_time}s")
		print(f"FPS: {params.average_sequence_length/(end_time-start_time)}")
