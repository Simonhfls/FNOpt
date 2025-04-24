from get_param import params,toCuda,toCpu,get_hyperparam,toDType
import matplotlib.pyplot as plt
from dataset_poisson import DatasetPoisson
#from setups_multistep_1_channel import Dataset
from dataset_utils import DatasetToSingleChannel
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from metamizer import get_Net
from Logger import Logger
import torch
import numpy as np
import time
import os
from derivatives import laplace,laplace_detach


logger = Logger(get_hyperparam(params),use_csv=False,use_tensorboard=False)

save = False#True#
if save:
	path = f"plots/{get_hyperparam(params).replace(' ','_').replace(';','_')}/poisson"
	os.makedirs(path,exist_ok=True)
	frame = 0

metamizer = toDType(toCuda(get_Net(params)))
#metamizer.nn = torch.compile(metamizer.nn)

device = 'cuda' if params.cuda else 'cpu'


date_time,index = logger.load_state(metamizer,None,datetime=params.load_date_time,index=params.load_index, device=device)
print(f"loaded: {date_time}, {index}")
metamizer.eval()

scales = []
#params.dt=0.1

investigate_steps = [0,1,2,3,20,50]
investigate_values = []

with torch.no_grad():#enable_grad():#
	for epoch in range(100):
		dataset = DatasetPoisson(params.height,params.width,1,1,params.average_sequence_length)
		dataset.reset1_env(0)
		#dataset.reset2_env(0) # 2 oppositely charged particles
		FPS=0
		start_time = time.time()

		for t in range(params.average_sequence_length):
			print(f"t: {t}")
			
			grads, hidden_states = dataset.ask()
			
			update_steps, new_hidden_states = metamizer(grads, hidden_states)
			#print(f"steps: {update_steps}")
			
			loss = dataset.tell(update_steps, new_hidden_states)
			
			scales.append(new_hidden_states[0][2][0,0,0,0].detach().cpu().numpy())
			
			if t in investigate_steps:
				
				investigate_values.append({"u":dataset.u[0,0].detach().cpu().numpy(),
							   "grad": grads[0,0].detach().cpu().numpy(),
							   "step": update_steps[0,0].detach().cpu().numpy(),
							   "scale": new_hidden_states[0][2][0,0,0,0].detach().cpu().numpy(),
							   "loss": loss.detach().cpu().numpy()})
			
			if t==investigate_steps[-1] and True:
				if False:
					dpi = 200
					fig = plt.figure(num=1,figsize=(2700/dpi,1000/dpi),dpi=dpi)
					plt.clf()
					for i,(step,values) in enumerate(zip(investigate_steps,investigate_values)):
						ax = fig.add_subplot(3,len(investigate_steps),1+i,projection='3d')
						bc_masks = dataset.bc_mask[0,0].cpu()
						u = values["u"]
						facecolors = cm.viridis((u - u.min()) / (u.max() - u.min()))  # Normalized colors
						facecolors[bc_masks>0.5] = [0, 0, 0, 1]
						
						surf = ax.plot_surface(X=dataset.x_mesh.cpu(),Y=dataset.y_mesh.cpu(),Z=u, facecolors=facecolors,linewidth=0.1, antialiased=False,shade=True,zorder=4)
						
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
						
						ax.set_zlim(-1.01, 1.01)
						ax.set_xlim(-1, params.height+1)
						ax.set_ylim(-1, params.height+1)
						ax.set_title(f"iteration: {step}")
						
						ax = fig.add_subplot(3,len(investigate_steps),1+i+len(investigate_steps))
						ax.set_xticks([])
						ax.set_yticks([])
						if i==0:
							ax.set_ylabel("$\\nabla L$")
						divider = make_axes_locatable(ax)
						cax = divider.append_axes('right', size='5%', pad=0.05)
						im = ax.imshow(values["grad"])
						cbar = fig.colorbar(im, cax=cax, orientation='vertical')
						cbar.formatter.set_powerlimits((0,0))
						
						ax = fig.add_subplot(3,len(investigate_steps),1+i+2*len(investigate_steps))
						ax.set_xticks([])
						ax.set_yticks([])
						if i==0:
							ax.set_ylabel("$\\Delta x$")
						divider = make_axes_locatable(ax)
						cax = divider.append_axes('right', size='5%', pad=0.05)
						im = ax.imshow(values["step"])
						cbar = fig.colorbar(im, cax=cax, orientation='vertical')
						cbar.formatter.set_powerlimits((0,0))
					
					if save:
						#plt.savefig(f"{path}/{str(frame).zfill(4)}.png",dpi=dpi)
						plt.savefig(f"{path}/laplace_steps.pdf",dpi=dpi, bbox_inches="tight")
						
					plt.show()
				pass
			
			# TODO: visualize, how gradient scaling changes during update steps
			
			if t%1==0:
				if False:
					plt.figure(3,figsize=(20,20),dpi=200)
					plt.clf()
					
					plt.subplot(2,3,1)
					plt.imshow(dataset.u[0,0].detach().cpu())
					plt.colorbar()
					plt.title("u")
					
					plt.subplot(2,3,2)
					plt.imshow(grads[0,0].detach().cpu())
					plt.colorbar()
					plt.title("grads")
					"""
					plt.subplot(2,3,3)
					plt.imshow(dataset.f[0,0].detach().cpu())
					plt.colorbar()
					plt.title("f")
					"""
					plt.subplot(2,3,3)
					plt.imshow(update_steps[0,0].detach().cpu())
					plt.colorbar()
					plt.title("update_steps")
					
					plt.subplot(2,3,4)
					plt.imshow(dataset.bc_mask[0,0].detach().cpu())
					plt.colorbar()
					plt.title("bc_mask")
					
					plt.subplot(2,3,5)
					plt.imshow((dataset.bc_values[0,0]*dataset.bc_mask[0,0]).detach().cpu())
					plt.colorbar()
					plt.title("bc_values")
					
					
					plt.subplot(2,3,6)
					u = dataset.bc_mask * dataset.bc_values + (1-dataset.bc_mask)*dataset.u
					residuals = laplace_detach(u)+dataset.f
					plt.imshow((residuals[0,0]*(1-dataset.bc_mask[0,0])).detach().cpu())
					plt.colorbar()
					plt.title("residuals")
					
					print(f"res: {torch.sum((residuals*(1-dataset.bc_mask))**2)/params.width/params.height} / loss: {loss}")
					
					plt.suptitle(f"t: {t}; loss: {loss}")
					
					plt.draw()
					plt.pause(0.001)
				
				if False:
					plt.figure(4,figsize=(20,20),dpi=200)
					plt.clf()
					
					plt.subplot(1,4,1)
					plt.imshow(dataset.u[0,0].detach().cpu())
					plt.colorbar()
					plt.title("u")
					
					plt.subplot(1,4,2)
					u = dataset.bc_mask * dataset.bc_values + (1-dataset.bc_mask)*dataset.u
					residuals = laplace_detach(u)+dataset.f
					plt.imshow((residuals[0,0]*(1-dataset.bc_mask[0,0])).detach().cpu())
					plt.colorbar()
					plt.title("residuals")
					
					plt.subplot(1,4,3)
					plt.imshow(grads[0,0].detach().cpu())
					plt.colorbar()
					plt.title("grads")
					
					plt.subplot(1,4,4)
					plt.imshow(update_steps[0,0].detach().cpu())
					plt.colorbar()
					plt.title("update_steps")
					
					
					print(f"res: {torch.sum((residuals*(1-dataset.bc_mask))**2)/params.width/params.height} / loss: {loss}")
					
					plt.suptitle(f"t: {t}; loss: {loss}")
					
					plt.draw()
					plt.pause(0.001)
					
				if False:
					dpi = 200
					plt.figure(2,figsize=(1200/dpi,600/dpi),dpi=dpi)
					plt.clf()
					plt.semilogy(scales)
					plt.xlabel("iteration")
					plt.ylabel("scale")
					plt.title("Laplace equation")
					
					plt.grid(True, which="major", ls="-", color='0.85')
					plt.grid(True, which="minor", ls="--", color='0.95')
					if save and t==60:
						plt.savefig(f"{path}/laplace_scaling.pdf",dpi=dpi,bbox_inches="tight")
						exit()
					
					plt.draw()
					plt.pause(0.001)
				
				if True:
					dpi = 200
					fig = plt.figure(num=5,figsize=(1400/dpi,1200/dpi),dpi=dpi)
					plt.clf()
					
					ax = fig.add_subplot(2,2,1,projection='3d')
					bc_masks = dataset.bc_mask[0,0].cpu()
					u = (dataset.bc_mask * dataset.bc_values + (1-dataset.bc_mask)*dataset.u)[0,0].detach().cpu()
					facecolors = cm.viridis((u - u.min()) / (u.max() - u.min()))  # Normalized colors
					facecolors[bc_masks>0.5] = [0, 0, 0, 1]
					
					surf = ax.plot_surface(X=dataset.x_mesh.cpu(),Y=dataset.y_mesh.cpu(),Z=u, facecolors=facecolors,linewidth=0.1, antialiased=False,shade=True,zorder=4)
					
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
					
					ax.set_zlim(-1.01, 1.01)
					ax.set_xlim(-1, params.height+1)
					ax.set_ylim(-1, params.height+1)
					ax.set_title(f"solution")
					
					ax = fig.add_subplot(2,2,2)
					ax.set_xticks([])
					ax.set_yticks([])
					ax.set_title("$\\nabla L$")
					divider = make_axes_locatable(ax)
					cax = divider.append_axes('right', size='5%', pad=0.05)
					im = ax.imshow(grads[0,0].detach().cpu())
					cbar = fig.colorbar(im, cax=cax, orientation='vertical')
					cbar.formatter.set_powerlimits((0,0))
					
					ax = fig.add_subplot(2,2,4)
					ax.set_xticks([])
					ax.set_yticks([])
					ax.set_title("$\\Delta x$")
					divider = make_axes_locatable(ax)
					cax = divider.append_axes('right', size='5%', pad=0.05)
					im = ax.imshow(update_steps[0,0].detach().cpu())
					cbar = fig.colorbar(im, cax=cax, orientation='vertical')
					cbar.formatter.set_powerlimits((0,0))
					
					ax = fig.add_subplot(2,2,3)
					ax.set_xlabel("iteration")
					ax.set_ylabel("$s_i$")
					ax.set_title("scaling")
					divider = make_axes_locatable(ax)
					ax.semilogy(scales)
					
				
					if save:
						plt.savefig(f"{path}/{str(frame).zfill(4)}.png",dpi=dpi,bbox_inches="tight")
						#plt.savefig(f"{path}/{str(frame).zfill(4)}.pdf",dpi=dpi)
						frame += 1
					
					plt.draw()
					plt.pause(0.001)
				
				if False:
					plt.figure(num=1,figsize=(20,20),dpi=200)
					plt.clf()
					fig, ax = plt.subplots(1,1,subplot_kw={"projection": "3d"},num=1,computed_zorder=False)
					
					
					bc_masks = dataset.bc_mask[0,0].cpu()
					u = dataset.u[0,0].detach().cpu()
					facecolors = cm.viridis((u - u.min()) / (u.max() - u.min()))  # Normalized colors
					facecolors[bc_masks>0.5] = [0, 0, 0, 1]
					
					surf = ax.plot_surface(X=dataset.x_mesh.cpu(),Y=dataset.y_mesh.cpu(),Z=u, facecolors=facecolors,linewidth=0.1, antialiased=False,shade=True,zorder=4)
					
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
					
					ax.set_zlim(-1.01, 1.01)
					ax.set_xlim(-1, params.height+1)
					ax.set_ylim(-1, params.height+1)
					plt.title(f"iteration: {t}")
					
					if False and t%50==0:
						plt.show()
					else:
						plt.draw()
						plt.pause(0.001)
		
		end_time = time.time()
		print(f"dt = {end_time-start_time}s")
		print(f"FPS: {params.average_sequence_length/(end_time-start_time)}")
