import matplotlib.pyplot as plt
from dataset_fluid import DatasetFluid
from dataset_utils import DatasetToSingleChannel
from metamizer import get_Net
from Logger import Logger
import torch
import numpy as np
from get_param import params,toCuda,toCpu,get_hyperparam,toDType
import time
import os
from derivatives import vector2HSV, rot_mac, staggered2normal, normal2staggered
from derivatives import dx,dy,laplace,dx_left,dx_right,dy_top,dy_bottom,map_vy2vx_top,map_vy2vx_bottom,map_vx2vy_left,map_vx2vy_right

#torch.manual_seed(1)
torch.set_num_threads(4)
#np.random.seed(6)

save_movie = False

use_cv2 = True#False#

if use_cv2:
	import cv2

	# setup opencv windows:
	cv2.namedWindow('legend',cv2.WINDOW_NORMAL) # legend for velocity field
	vector = toCuda(torch.cat([torch.arange(-1,1,0.01).unsqueeze(0).unsqueeze(2).repeat(1,1,200),torch.arange(-1,1,0.01).unsqueeze(0).unsqueeze(1).repeat(1,200,1)]))
	image = vector2HSV(vector).astype(np.float32)
	image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
	cv2.imshow('legend',image)
	
	
	cv2.namedWindow('a',cv2.WINDOW_NORMAL)
	cv2.namedWindow('v',cv2.WINDOW_NORMAL)
	cv2.namedWindow('v_cond',cv2.WINDOW_NORMAL)
	cv2.namedWindow('p',cv2.WINDOW_NORMAL)
	
	
	# Mouse interactions:
	def mousePosition(event,x,y,flags,param):
		global original_dataset
		if (event==cv2.EVENT_MOUSEMOVE or event==cv2.EVENT_LBUTTONDOWN) and flags==1:
			original_dataset.mousex = x
			original_dataset.mousey = y
	cv2.setMouseCallback("p",mousePosition)
	cv2.setMouseCallback("v",mousePosition)
	cv2.setMouseCallback("a",mousePosition)


logger = Logger(get_hyperparam(params),use_csv=False,use_tensorboard=False)

params.dt = 4 # for equal comparison of loss with previous work, dt must be set to 4
dt = params.dt

save = False#True#
if save:
	path = f"plots/{get_hyperparam(params).replace(' ','_').replace(';','_')}/fluid/mu_{params.mu} rho_{params.rho}"
	os.makedirs(path,exist_ok=True)
	frame = 0

metamizer = toDType(toCuda(get_Net(params)))

device = 'cuda' if params.cuda else 'cpu'


date_time,index = logger.load_state(metamizer,None,datetime=params.load_date_time,index=params.load_index, device=device)
print(f"loaded: {date_time}, {index}")
metamizer.eval()

scales = []
max_scales = []
L_ds = []
L_ps = []

with torch.no_grad():
	for epoch in range(10):
		original_dataset = DatasetFluid(params.height,params.width,1,1,params.average_sequence_length,iterations_per_timestep=params.iterations_per_timestep,interactive=True,types=["simple_benchmark"])#,"magnus","box"
		
		original_dataset.mus[:] = params.mu#0.1 # 5 # 
		original_dataset.rhos[:] = params.rho#4 # 1 #
		
		original_dataset.bc_mask[:] = 1
		original_dataset.bc_mask[:,:,3:(params.height-3),5:(params.width-5)]=0
		object_x=25
		object_y=50
		object_w=5#10
		object_h=5#15
		original_dataset.bc_mask[:,:,(object_y-object_h):(object_y+object_h),(object_x-object_w):(object_x+object_w)] = 1
		
		dataset = DatasetToSingleChannel(original_dataset)
		FPS=0
		start_time = time.time()
		
		for t in range(params.average_sequence_length*params.iterations_per_timestep):
			print(f"t: {t}")
			
			grads, hidden_states = dataset.ask()
			
			update_steps, new_hidden_states = metamizer(grads, hidden_states)
			
			loss = dataset.tell(update_steps, new_hidden_states)
			
			scales.append(new_hidden_states[0][2][0,0,0,0].detach().cpu().numpy())
			
			if use_cv2:
				if t%params.iterations_per_timestep == 0:
					a_new = original_dataset.a_new[0][0]
					p = original_dataset.p[0][0]
					bc_mask = original_dataset.bc_mask[0][0]
					
					# print out a:
					a = a_new.clone().cpu()
					a = a-torch.min(a)
					a = toCpu(a/torch.max(a)).unsqueeze(2).repeat(1,1,3).numpy()
					if save_movie:
						movie_a.write((255*a).astype(np.uint8))
					cv2.imshow('a',a)
					
					# print out v:
					
					#v_new = flow_mask_mac*v_new+cond_mask_mac*v_cond
					v_new = rot_mac(a_new.clone().unsqueeze(0).unsqueeze(1))
					vector = staggered2normal(v_new.clone())[0,:,2:-1,2:-1]
					image = vector2HSV(vector).astype(np.float32)
					image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
					
					
					if save_movie:
						movie_v.write((255*image).astype(np.uint8))
					cv2.imshow('v',image)
					
					
					cond_mask_mac = (normal2staggered(original_dataset.bc_mask.repeat(1,2,1,1))==1).float()
					v_cond = original_dataset.bc_values.clone()
					v_cond = normal2staggered(v_cond)
					v_cond = cond_mask_mac*v_cond
					image = vector2HSV(v_cond[0]).astype(np.float32)
					image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
					cv2.imshow('v_cond',image)
					
					
					# print out p:
					p = p.clone()
					p = p*(1-bc_mask)
					p = p + bc_mask * torch.sum(p)/torch.sum(1-bc_mask)
					#p = flow_mask[0,0]*p_new[0,0].clone()
					p = p-torch.min(p)
					p = p/torch.max(p)
					p = toCpu(p).unsqueeze(2).repeat(1,1,3).numpy().astype(np.float32)
					if save_movie:
						movie_p.write((255*p).astype(np.uint8))
					cv2.imshow('p',p)
					
					# keyboard interactions:
					key = cv2.waitKey(1)
					
					
					if key==ord('1'): # change viscosity
						original_dataset.mus[:] /= 1.1
					if key==ord('2'):
						original_dataset.mus[:] *= 1.1
					if key==ord('3'): # change density
						original_dataset.rhos[:] /= 1.1
					if key==ord('4'):
						original_dataset.rhos[:] *= 1.1
					
					
					if key==ord('x'): # increase flow speed
						original_dataset.mousev+=0.1
					if key==ord('y'): # decrease flow speed
						original_dataset.mousev-=0.1
					
					if key==ord('s'): # increase angular velocity
						original_dataset.mousew+=0.1
					if key==ord('a'): # decrease angular velocity
						original_dataset.mousew-=0.1
					
					if key==ord('n'): # new simulation
						break
					if key==ord('q'): # quit simulation
						exit()
			else:
				
				if (t+1)%(params.iterations_per_timestep) == 0:
					a_new = original_dataset.a_new
					bc_mask = original_dataset.bc_mask
					p = original_dataset.p
					p = p.clone()
					p = p*(1-bc_mask)
					p = p + bc_mask * torch.sum(p)/torch.sum(1-bc_mask)
					v_new = rot_mac(a_new.clone())
					v_new_normal = staggered2normal(v_new.clone())[:,:,2:-1,2:-1]
					
					p_new = original_dataset.p
					rho = original_dataset.rhos
					mu = original_dataset.mus
					a_old = original_dataset.a_old
					v_old = rot_mac(a_old)
					cond_mask_mac = (normal2staggered(bc_mask.repeat(1,2,1,1))==1).float()
					flow_mask_mac = 1-cond_mask_mac
					# explicit / implicit / IMEX integration schemes
					if params.integrator == "explicit":
						v = v_old
					if params.integrator == "implicit":
						v = v_new
					if params.integrator == "imex":
						v = (v_new+v_old)/2
					
					res_p_u =  (flow_mask_mac[:,1:2]*((rho*((v_new[:,1:2]-v_old[:,1:2])/dt+v[:,1:2]*dx(v[:,1:2])+0.5*(map_vy2vx_top(v[:,0:1])*dy_top(v[:,1:2])+map_vy2vx_bottom(v[:,0:1])*dy_bottom(v[:,1:2])))+dx_left(p_new)-mu*laplace(v[:,1:2]))))[:,:,1:-1,1:-1]
					res_p_v = (flow_mask_mac[:,0:1]*((rho*((v_new[:,0:1]-v_old[:,0:1])/dt+v[:,0:1]*dy(v[:,0:1])+0.5*(map_vx2vy_left(v[:,1:2])*dx_left(v[:,0:1])+map_vx2vy_right(v[:,1:2])*dx_right(v[:,0:1])))+dy_top(p_new)-mu*laplace(v[:,0:1]))))[:,:,1:-1,1:-1]
					L_ps.append((torch.mean(res_p_u**2)+torch.mean(res_p_v**2)).detach().cpu().numpy())
					
					v_cond = original_dataset.bc_values.clone()
					v_cond = normal2staggered(v_cond)
					v_new = cond_mask_mac*v_cond + flow_mask_mac*v_new
					res_d = (dx_right(v_new[:,1:2])+dy_bottom(v_new[:,0:1]))[:,:,2:-2,2:-2]
					L_ds.append(torch.mean(res_d**2).detach().cpu().numpy())
					
					if False:
						plt.figure(1)
						plt.clf()
						
						plt.subplot(2,4,1)
						plt.imshow(a_new[0,0].detach().cpu())
						plt.colorbar()
						plt.title("a_new")
						
						plt.subplot(2,4,2)
						plt.imshow(grads[0,0].detach().cpu())
						plt.colorbar()
						plt.title("a_grad")
						
						plt.subplot(2,4,5)
						plt.imshow(p[0,0].detach().cpu())
						plt.colorbar()
						plt.title("p")
						
						plt.subplot(2,4,6)
						plt.imshow(grads[1,0].detach().cpu())
						plt.colorbar()
						plt.title("p_grad")
						
						plt.subplot(2,4,3)
						plt.imshow(v_new_normal[0,0].detach().cpu())
						plt.colorbar()
						plt.title("u")
						
						"""
						plt.subplot(2,4,7)
						plt.imshow(v_new_normal[0,1].detach().cpu())
						plt.colorbar()
						plt.title("v")
						"""
						
						plt.subplot(2,4,4)
						plt.imshow(res_p_u[0,0].detach().cpu())
						plt.colorbar()
						plt.title("res_p_u")
						
						plt.subplot(2,4,8)
						plt.imshow(res_p_v[0,0].detach().cpu())
						plt.colorbar()
						plt.title("res_p_v")
						
						plt.subplot(2,4,7)
						plt.imshow(res_d[0,0].detach().cpu())
						plt.colorbar()
						plt.title("res_d")
						
						print(f"L_d: {torch.mean(res_d**2)} / L_p: {torch.mean(res_p_u**2)+torch.mean(res_p_v**2)} /")
						
						plt.draw()
						plt.pause(0.01)
						
					if True:
						dpi=200
						plt.figure(5,figsize=(1200/dpi,1200/dpi),dpi=dpi)
						plt.clf()
						
						plt.subplot(2,2,1)
						plt.imshow(a_new[0,0].detach().cpu())
						plt.colorbar()
						plt.title("a")
						plt.axis('off')
						
						plt.subplot(2,2,2)
						plt.imshow(p[0,0].detach().cpu())
						plt.colorbar()
						plt.title("p")
						plt.axis('off')
						
						plt.subplot(2,2,3)
						plt.imshow(v_new_normal[0,0].detach().cpu())
						plt.colorbar()
						plt.title("u")
						plt.axis('off')
						
						plt.subplot(2,2,4)
						plt.imshow(v_new_normal[0,1].detach().cpu())
						plt.colorbar()
						plt.title("v")
						plt.axis('off')
						
						if save:
							plt.savefig(f"{path}/{str(frame).zfill(4)}.png",dpi=dpi)
							frame += 1
						
						plt.draw()
						plt.pause(0.01)
						
					if True:
						plt.figure(2)
						plt.clf()
						plt.semilogy(scales[-500:])
						plt.xlabel("iteration")
						plt.ylabel("scale")
						plt.legend(["scales"])
						
						plt.suptitle(f"timestep: {original_dataset.T[0].cpu().numpy()[0]}")
						
						plt.draw()
						plt.pause(0.01)
					
					if False:
						plt.figure(3)
						plt.clf()
						#fig, ax = plt.subplots(3)
						
						v_new = rot_mac(a_new.clone())
						flow = staggered2normal(v_new.clone())[0,:,2:-1,2:-1]
						image = vector2HSV(flow)
						
						flow = toCpu(flow).numpy()
						Y,X = np.mgrid[0:flow.shape[1],0:flow.shape[2]]
						linewidth = image[:,:,2]/np.max(image[:,:,2])
						plt.streamplot(X, Y, flow[1], flow[0], color='k', density=1,linewidth=2*linewidth)
						palette = plt.cm.viridis#gnuplot2
						palette.set_bad('k',1.0)
						pm = np.ma.masked_where(toCpu(bc_mask).numpy()==1, toCpu(p).numpy())
						plt.imshow(pm[0,0,2:-1,2:-1],cmap=palette)
						plt.axis('off')
						
						"""
						plt.imshow(p[0,0].detach().cpu())
						plt.colorbar()
						plt.axis('off')"""
						plt.title(f"timestep: {original_dataset.T[0].cpu().numpy()[0]}")
						plt.draw()
						plt.pause(0.01)
