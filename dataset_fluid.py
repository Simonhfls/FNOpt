import torch
import numpy as np
from get_param2 import params,toCuda,toCpu,device
from derivatives import dx,dy,dx_left,dy_top,dx_right,dy_bottom,laplace,laplace_detach,map_vx2vy_left,map_vy2vx_top,map_vx2vy_right,map_vy2vx_bottom,normal2staggered,rot_mac
from utils import normalize_grads
from PIL import Image
from utils import log_range_params,has_nan,has_inf,value_range

# we can define domain boundaries inside these .png images.
# These images were not taken into account during training to test the generalization performance of our models.
cyber_truck = toCuda(torch.mean(torch.tensor(np.asarray(Image.open('imgs/cyber.png'))).float(),dim=2)<100).float()
fish = toCuda(torch.mean(torch.tensor(np.asarray(Image.open('imgs/fish.png'))).float(),dim=2)<100).float()
smiley = toCuda(torch.mean(torch.tensor(np.asarray(Image.open('imgs/smiley.png'))).float(),dim=2)<100).float()
wing = toCuda(torch.mean(torch.tensor(np.asarray(Image.open('imgs/wing_profile.png'))).float(),dim=2)<100).float()
background1 = toCuda(torch.mean(torch.tensor(np.asarray(Image.open('imgs/background1.png'))).float(),dim=2)<100).float()
background2 = toCuda(torch.mean(torch.tensor(np.asarray(Image.open('imgs/background2.png'))).float(),dim=2)<100).float()
background3 = toCuda(torch.mean(torch.tensor(np.asarray(Image.open('imgs/background3.png'))).float(),dim=2)<100).float()

images = {"cyber":cyber_truck, "fish":fish, "smiley":smiley, "wing":wing}
backgrounds = {"empty":background1,"cave1":background2,"cave2":background3}

#Attention: x/y are swapped (x-dimension=1; y-dimension=0)

def loss(a_old,a_new,p_new,bc_mask,bc_values,rho=1,mu=1,dt=params.cloth.dt):
	"""
	rho = mu = dt = 1
	
	nan_here = False
	for value,name in zip([a_old,a_new,p_new,bc_mask,bc_values,rho,mu,dt],["a_old","a_new","p_new","bc_mask","bc_values","rho","mu","dt"]):
		print(f"input {name}: {value_range(value)}")
		if has_nan(value):
			print(f"loss input {name} has nan!")
			nan_here = True
		if has_inf(value):
			print(f"loss input {name} has inf!")
	"""
	
	dt = 4 # TODO: parameterize dt
	
	# compute dirichlet boundary values / mask on MAC grid
	bc_values = normal2staggered(bc_values)
	cond_mask_mac = (normal2staggered(bc_mask.repeat(1,2,1,1))==1).float()
	flow_mask_mac = 1-cond_mask_mac
	
	# obtain divergence free velocity field from vector potential
	v_old = rot_mac(a_old)
	v_new = rot_mac(a_new)
	
	# compute boundary loss
	loss_bound = torch.mean((cond_mask_mac*(v_new-bc_values)**2)[:,:,1:-1,1:-1],dim=(1,2,3))
	
	# explicit / implicit / IMEX integration schemes
	if params.Fluid.integrator == "explicit":
		v = v_old
	if params.Fluid.integrator == "implicit":
		v = v_new
	if params.Fluid.integrator == "imex":
		v = (v_new+v_old)/2
	
	# compute loss for momentum equation
	loss_nav =  torch.mean((flow_mask_mac[:,1:2]*((rho*((v_new[:,1:2]-v_old[:,1:2])/dt+v[:,1:2]*dx(v[:,1:2])+0.5*(map_vy2vx_top(v[:,0:1])*dy_top(v[:,1:2])+map_vy2vx_bottom(v[:,0:1])*dy_bottom(v[:,1:2])))+dx_left(p_new)-mu*laplace(v[:,1:2])))**2)[:,:,1:-1,1:-1],dim=(1,2,3))+\
			torch.mean((flow_mask_mac[:,0:1]*((rho*((v_new[:,0:1]-v_old[:,0:1])/dt+v[:,0:1]*dy(v[:,0:1])+0.5*(map_vx2vy_left(v[:,1:2])*dx_left(v[:,0:1])+map_vx2vy_right(v[:,1:2])*dx_right(v[:,0:1])))+dy_top(p_new)-mu*laplace(v[:,0:1])))**2)[:,:,1:-1,1:-1],dim=(1,2,3))
	#loss_nav =  torch.mean((flow_mask_mac[:,1:2]*((rho*((v_new[:,1:2]-v_old[:,1:2])/dt+v[:,1:2]*dx(v[:,1:2])+0.5*(map_vy2vx_top(v[:,0:1])*dy_top(v[:,1:2])+map_vy2vx_bottom(v[:,0:1])*dy_bottom(v[:,1:2])))+dx_left(p_new)-mu*laplace_detach(v[:,1:2])))**2)[:,:,1:-1,1:-1],dim=(1,2,3))+\
	#		torch.mean((flow_mask_mac[:,0:1]*((rho*((v_new[:,0:1]-v_old[:,0:1])/dt+v[:,0:1]*dy(v[:,0:1])+0.5*(map_vx2vy_left(v[:,1:2])*dx_left(v[:,0:1])+map_vx2vy_right(v[:,1:2])*dx_right(v[:,0:1])))+dy_top(p_new)-mu*laplace_detach(v[:,0:1])))**2)[:,:,1:-1,1:-1],dim=(1,2,3))
	#loss_nav =  torch.mean((flow_mask_mac[:,1:2]*((rho*((v_new[:,1:2]-v_old[:,1:2])/dt+v[:,1:2]*dx(v[:,1:2].detach())+0.5*(map_vy2vx_top(v[:,0:1])*dy_top(v[:,1:2].detach())+map_vy2vx_bottom(v[:,0:1])*dy_bottom(v[:,1:2].detach())))+dx_left(p_new)-mu*laplace_detach(v[:,1:2])))**2)[:,:,1:-1,1:-1],dim=(1,2,3))+\
	#		torch.mean((flow_mask_mac[:,0:1]*((rho*((v_new[:,0:1]-v_old[:,0:1])/dt+v[:,0:1]*dy(v[:,0:1].detach())+0.5*(map_vx2vy_left(v[:,1:2])*dx_left(v[:,0:1].detach())+map_vx2vy_right(v[:,1:2])*dx_right(v[:,0:1].detach())))+dy_top(p_new)-mu*laplace_detach(v[:,0:1])))**2)[:,:,1:-1,1:-1],dim=(1,2,3))
	
	# regularize gradients of pressure field (for very high reynolds numbers)
	regularize_grad_p = torch.mean((dx_right(p_new)**2+dy_bottom(p_new)**2)[:,:,2:-2,2:-2],dim=(1,2,3))
	
	# optional: additional loss to keep mean of a / p close to 0
	loss_mean_a = torch.mean(a_new,dim=(1,2,3))**2
	loss_mean_p = torch.mean(p_new,dim=(1,2,3))**2
	
	loss = normalize_grads(params.Fluid.loss_bound*loss_bound) + normalize_grads(params.Fluid.loss_nav*loss_nav)
	
	if params.Fluid.loss_mean_a:
		loss = loss + normalize_grads(params.Fluid.loss_mean_a*loss_mean_a)
	if params.Fluid.loss_mean_p:
		loss = loss + normalize_grads(params.Fluid.loss_mean_p*loss_mean_p)
	if params.Fluid.regularize_grad_p:
		loss = loss + normalize_grads(params.Fluid.regularize_grad_p*regularize_grad_p)
	
	"""
	for value,name in zip([bc_values,cond_mask_mac,flow_mask_mac,v_old,v_new,loss_bound,v,loss_nav,regularize_grad_p,loss_mean_a,loss_mean_p,loss],["bc_values","cond_mask_mac","flow_mask_mac","v_old","v_new","loss_bound","v","loss_nav","regularize_grad_p","loss_mean_a","loss_mean_p","loss"]):
		print(f"output {name}: {value_range(value)}")
		if has_nan(value):
			print(f"loss output {name} has nan!")
			nan_here = True
		if has_nan(value):
			print(f"loss output {name} has inf!")
	
	if nan_here:
		exit()
	"""
	return loss

class DatasetFluid:
	def __init__(self,h,w,batch_size=100,dataset_size=1000,average_sequence_length=5000,interactive=False,max_speed=3,brown_damping=0.9995,brown_velocity=0.005,init_velocity=0,dt=1,types=["magnus","box","pipe"],images=["cyber","fish","smiley","wing"],background_images=["empty"],iterations_per_timestep=10,mu_range=None,rho_range=None):
		"""
		create dataset
		:h: height of domains
		:w: width of domains
		:batch_size: batch_size for ask()
		:dataset_size: size of dataset
		:average_sequence_length: average length of sequence until domain gets reset
		:interactive: allows to interact with the dataset by changing the following variables:
			- mousex: x-position of obstacle
			- mousey: y-position of obstacle
			- mousev: velocity of fluid
			- mousew: angular velocity of ball
		:max_speed: maximum speed at dirichlet boundary conditions
		:brown_damping / brown_velocity: parameters for random motions in dataset
		:init_velocity: initial velocity of objects in simulation (can be ignored / set to 0)
		:dt: time step size of simulation
		:types: list of environments that can be chosen from:
			- "magnus": train magnus effect on randomly moving / rotating balls of random radia
			- "box": train with randomly moving boxes of random width / height
			- "pipe": difficult pipe-environment that contains long range dependencies
			- "image": choose a random image from images as moving obstacle
		:images: list of images that can be chosen from, if 'image' is contained in types-list. You can simply add more images by adding them to the global images-dictionary.
		:background_images: you can also change the static background if the image-type is chosen.
		"""
		#CODO: add more different environemts; add neumann boundary conditions
		self.h,self.w = h,w
		self.batch_size = batch_size
		self.dataset_size = dataset_size
		self.average_sequence_length = average_sequence_length
		self.a_old = toCuda(torch.zeros(dataset_size,1,h,w))
		self.a_new = toCuda(torch.zeros(dataset_size,1,h,w))
		self.p = toCuda(torch.zeros(dataset_size,1,h,w))
		self.bc_values = toCuda(torch.zeros(dataset_size,2,h,w))# one could also think about p_cond... -> neumann
		self.bc_mask = toCuda(torch.zeros(dataset_size,1,h,w))
		self.T = toCuda(torch.zeros(self.dataset_size,1)) # timestep
		self.iterations = toCuda(torch.zeros(dataset_size)) # iterations for the individual training pool samples
		self.iterations_per_timestep = iterations_per_timestep # number of iterations per timestep
		
		self.hidden_states = [None for _ in range(dataset_size)]
		
		self.padding_x,self.padding_y = 5,3
		
		mu_range = [params.Fluid.min_mu,params.Fluid.mu] if mu_range is None else mu_range
		rho_range = [params.Fluid.min_rho,params.Fluid.rho] if rho_range is None else rho_range
		self.mu_range = log_range_params(mu_range)
		self.rho_range = log_range_params(rho_range)
		#self.mus = toCuda(torch.zeros(self.dataset_size,1,1,1))
		#self.rhos = toCuda(torch.zeros(self.dataset_size,1,1,1))
		
		self.mus = toCuda(torch.exp(self.mu_range[0]+torch.rand(self.dataset_size,1,1,1)*self.mu_range[1]))#1000# # => scheint nicht die problem-quelle zu sein
		self.rhos = toCuda(torch.exp(self.rho_range[0]+torch.rand(self.dataset_size,1,1,1)*self.rho_range[1]))#10#0#
		
		self.env_info = [{} for _ in range(dataset_size)]
		self.interactive = interactive
		self.interactive_spring = 150#300#200#~ 1/spring constant to move object
		self.max_speed = max_speed
		self.brown_damping = brown_damping
		self.brown_velocity = brown_velocity
		self.init_velocity = init_velocity
		
		self.dt = dt
		self.types = types
		self.images = images
		self.background_images = background_images
			
		self.mousex = 0
		self.mousey = 0
		self.mousev = 0
		self.mousew = 0
		
		for i in range(dataset_size):
			self.reset_env(i)
		
		self.step = 0 # number of tell()-calls
		self.reset_i = 0
		
	def reset_env(self,index):
		"""
		reset environemt[index] to a new, randomly chosen domain
		a and p are set to 0, so the model has to learn "cold-starts"
		"""
		
		# reset state
		self.a_old[index,:,:,:] = 0
		self.a_new[index,:,:,:] = 0
		self.p[index,:,:,:] = 0
		self.hidden_states[index] = None
		self.T[index] = 0
		self.iterations[index] = 0
		
		
		# material parameters
		#self.mus[index] = torch.exp(self.mu_range[0]+torch.rand(1)*self.mu_range[1])#1000# # => scheint nicht die problem-quelle zu sein
		#self.rhos[index] = torch.exp(self.rho_range[0]+torch.rand(1)*self.rho_range[1])#10#0#
		
		# boundary conditions
		self.bc_mask[index,:,:,:]=0
		self.bc_mask[index,:,0:3,:]=1
		self.bc_mask[index,:,(self.h-3):self.h,:]=1
		self.bc_mask[index,:,:,0:5]=1
		self.bc_mask[index,:,:,(self.w-5):self.w]=1
		
		type = np.random.choice(self.types)

		if type == "magnus": # magnus effekt (1)
			flow_v = self.max_speed*(np.random.rand()-0.5)*2 #flow velocity (1.5) (before: 3*(np.random.rand()-0.5)*2)
			object_y = np.random.randint(self.h//2-10,self.h//2+10)
			#CODO: implement this in a more elegant way by flipping environment
			if flow_v>0:
				object_x = np.random.randint(self.w//4-10,self.w//4+10)
			else:
				object_x = np.random.randint(3*self.w//4-10,3*self.w//4+10)
			object_x = np.random.randint(self.w//2-10,self.w//2+10)
			object_vx = self.init_velocity*(np.random.rand()-0.5)*2
			object_vy = self.init_velocity*(np.random.rand()-0.5)*2
			object_r = np.random.randint(5,20) # object radius (15)
			object_w = self.max_speed*(np.random.rand()-0.5)*2/object_r # object angular velocity (3/object_r)
			
			# 1. generate mesh 2 x [2r x 2r]
			y_mesh,x_mesh = toCuda(torch.meshgrid([torch.arange(-object_r,object_r+1),torch.arange(-object_r,object_r+1)]))
			
			# 2. generate mask
			mask_ball = ((x_mesh**2+y_mesh**2)<object_r**2).float().unsqueeze(0)
			
			# 3. generate bc_values and multiply with mask
			v_ball = object_w*torch.cat([x_mesh.unsqueeze(0),-y_mesh.unsqueeze(0)])*mask_ball
			
			# 4. add masks / bc_values
			#print(f"shapes: {self.bc_mask[index,:,(object_y-object_r):(object_y+object_r+1),(object_x-object_r):(object_x+object_r+1)].shape} vs {mask_ball.shape}")
			self.bc_mask[index,:,(object_y-object_r):(object_y+object_r+1),(object_x-object_r):(object_x+object_r+1)] += mask_ball
			self.bc_values[index,0,(object_y-object_r):(object_y+object_r+1),(object_x-object_r):(object_x+object_r+1)] += v_ball[0]+object_vy
			self.bc_values[index,1,(object_y-object_r):(object_y+object_r+1),(object_x-object_r):(object_x+object_r+1)] += v_ball[1]+object_vx
			
			
			self.bc_values[index,1,10:(self.h-10),0:5]=flow_v
			self.bc_values[index,1,10:(self.h-10),(self.w-5):self.w]=flow_v
			
			self.env_info[index]["type"] = type
			self.env_info[index]["x"] = object_x
			self.env_info[index]["y"] = object_y
			self.env_info[index]["vx"] = object_vx
			self.env_info[index]["vy"] = object_vy
			self.env_info[index]["r"] = object_r
			self.env_info[index]["w"] = object_w
			self.env_info[index]["flow_v"] = flow_v
			self.mousex = object_x
			self.mousey = object_y
			self.mousev = flow_v
			self.mousew = object_w*object_r
			
		if type == "DFG_benchmark": # DFG benchmark setup from: http://www.featflow.de/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html
			flow_v = self.max_speed*(np.random.rand()-0.5)*2 #flow velocity TODO: set to 0.3 / 1.5
			object_r = 0.05/0.41*(self.h-2*self.padding_y) # object radius
			
			object_y = 0.2/0.41*(self.h-2*self.padding_y)+self.padding_y
			object_x = 0.2/0.41*(self.h-2*self.padding_y)+self.padding_x
			
			object_vx,object_vy,object_w = 0,0,0 # object angular velocity
			
			# 1. generate mesh 2 x [2r x 2r]
			y_mesh,x_mesh = toCuda(torch.meshgrid([torch.arange(-object_r,object_r+1),torch.arange(-object_r,object_r+1)]))
			
			# 2. generate mask
			mask_ball = ((x_mesh**2+y_mesh**2)<object_r**2).float().unsqueeze(0)
			
			# 3. generate bc_values and multiply with mask
			v_ball = object_w*torch.cat([x_mesh.unsqueeze(0),-y_mesh.unsqueeze(0)])*mask_ball
			
			# 4. add masks / bc_values
			x_pos1, y_pos1 = int((object_x-object_r)),int((object_y-object_r))
			x_pos2, y_pos2 = x_pos1+mask_ball.shape[1],y_pos1+mask_ball.shape[2]
			self.bc_mask[index,:,y_pos1:y_pos2,x_pos1:x_pos2] += mask_ball
			self.bc_values[index,0,y_pos1:y_pos2,x_pos1:x_pos2] += v_ball[0]+object_vy
			self.bc_values[index,1,y_pos1:y_pos2,x_pos1:x_pos2] += v_ball[1]+object_vx
			
			# inlet / outlet flow
			profile_size = self.bc_values[index,0,(self.padding_y):-(self.padding_y),:(self.padding_x)].shape[0]
			flow_profile = toCuda(torch.arange(0,profile_size,1.0))
			flow_profile *= 0.41/flow_profile[-1]
			flow_profile = 4*flow_profile*(0.41-flow_profile)/0.1681
			flow_profile = flow_profile.unsqueeze(1)
			self.bc_values[index,1,(self.padding_y):-(self.padding_y),:(self.padding_x)] = flow_v*flow_profile
			self.bc_values[index,1,(self.padding_y):-(self.padding_y),-(self.padding_x):] = flow_v*flow_profile
			
			self.env_info[index]["type"] = type
			self.env_info[index]["x"] = object_x
			self.env_info[index]["y"] = object_y
			self.env_info[index]["vx"] = object_vx
			self.env_info[index]["vy"] = object_vy
			self.env_info[index]["r"] = object_r
			self.env_info[index]["w"] = object_w
			self.env_info[index]["flow_v"] = flow_v
			self.mousex = object_x
			self.mousey = object_y
			self.mousev = flow_v
			self.mousew = object_w*object_r
			
		if type == "box":# block at random position
			object_h = np.random.randint(5,20) # object height / 2
			object_w = np.random.randint(5,20) # object width / 2
			flow_v = self.max_speed*(np.random.rand()-0.5)*2
			object_y = np.random.randint(self.h//2-10,self.h//2+10)
			if flow_v>0:
				object_x = np.random.randint(self.w//4-10,self.w//4+10)
			else:
				object_x = np.random.randint(3*self.w//4-10,3*self.w//4+10)
			object_vx = self.init_velocity*(np.random.rand()-0.5)*2
			object_vy = self.init_velocity*(np.random.rand()-0.5)*2
			
			self.bc_mask[index,:,(object_y-object_h):(object_y+object_h),(object_x-object_w):(object_x+object_w)] = 1
			self.bc_values[index,0,(object_y-object_h):(object_y+object_h),(object_x-object_w):(object_x+object_w)] = object_vy
			self.bc_values[index,1,(object_y-object_h):(object_y+object_h),(object_x-object_w):(object_x+object_w)] = object_vx
			
			self.bc_values[index,1,10:(self.h-10),0:5]=flow_v
			self.bc_values[index,1,10:(self.h-10),(self.w-5):self.w]=flow_v
			
			self.env_info[index]["type"] = type
			self.env_info[index]["x"] = object_x
			self.env_info[index]["y"] = object_y
			self.env_info[index]["vx"] = object_vx
			self.env_info[index]["vy"] = object_vy
			self.env_info[index]["h"] = object_h
			self.env_info[index]["w"] = object_w
			self.env_info[index]["flow_v"] = flow_v
			self.mousex = object_x
			self.mousey = object_y
			self.mousev = flow_v
			
		if type == "pipe":# "pipes-labyrinth"
			flow_v = self.max_speed*(np.random.rand()-0.5)*2
			self.bc_values[index,1,10:(self.h//4),0:5]=flow_v
			self.bc_values[index,1,(3*self.h//4):(self.h-10),(self.w-5):self.w]=flow_v
			
			self.bc_mask[index,:,(self.h//3-2):(self.h//3+2),0:(3*self.w//4)] = 1
			self.bc_mask[index,:,(2*self.h//3-2):(2*self.h//3+2),(self.w//4):self.w] = 1
			if np.random.rand()<0.5:
				self.bc_mask[index] = self.bc_mask[index].flip(1)
				self.bc_values[index] = self.bc_values[index].flip(1)
			
			self.env_info[index]["type"] = type
			self.env_info[index]["flow_v"] = flow_v
			self.mousev = flow_v
		
		if type == "image":
			flow_v = self.max_speed*(np.random.rand()-0.5)*2
			object_y = np.random.randint(self.h//2-10,self.h//2+10)
			if flow_v>0:
				object_x = np.random.randint(self.w//4-10,self.w//4+10)
			else:
				object_x = np.random.randint(3*self.w//4-10,3*self.w//4+10)
			object_vx = self.init_velocity*(np.random.rand()-0.5)*2
			object_vy = self.init_velocity*(np.random.rand()-0.5)*2
			
			image = np.random.choice(self.images)
			image_mask = images[image]
			object_h, object_w = image_mask.shape[0], image_mask.shape[1]
			background_image = np.random.choice(self.background_images)
			background_image_mask = backgrounds[background_image]
			
			self.bc_mask[index,:] = 1-(1-self.bc_mask[index,:])*(1-background_image_mask)
			self.bc_mask[index,:,(object_y-object_h//2):(object_y-object_h//2+object_h),(object_x-object_w//2):(object_x-object_w//2+object_w)] = 1-(1-self.bc_mask[index,:,(object_y-object_h//2):(object_y-object_h//2+object_h),(object_x-object_w//2):(object_x-object_w//2+object_w)])*(1-image_mask)
			self.bc_values[index,:]=0
			self.bc_values[index,1,20:(self.h-20),0:5]=flow_v
			self.bc_values[index,1,20:(self.h-20),(self.w-5):self.w]=flow_v
			self.bc_values[index,0:1,(object_y-object_h//2):(object_y-object_h//2+object_h),(object_x-object_w//2):(object_x-object_w//2+object_w)] = object_vy*image_mask
			self.bc_values[index,1:2,(object_y-object_h//2):(object_y-object_h//2+object_h),(object_x-object_w//2):(object_x-object_w//2+object_w)] = object_vx*image_mask
			
			self.env_info[index]["type"] = type
			self.env_info[index]["image"] = image
			self.env_info[index]["background_image"] = background_image
			self.env_info[index]["x"] = object_x
			self.env_info[index]["y"] = object_y
			self.env_info[index]["vx"] = object_vx
			self.env_info[index]["vy"] = object_vy
			self.env_info[index]["h"] = object_h
			self.env_info[index]["w"] = object_w
			self.env_info[index]["flow_v"] = flow_v
			self.mousex = object_x
			self.mousey = object_y
			self.mousev = flow_v
		
		if type == "simple_benchmark": # simple banchmark from first paper
			
			self.mus[index] = 0.1
			self.rhos[index] = 1
			
			# set boundary conditions
			object_x=50
			object_y=50
			object_w=5#10
			object_h=5#15
			
			self.bc_values[index] = 0
			self.bc_values[index,1,10:(self.h-10),0:5]=0.5
			self.bc_values[index,1,10:(self.h-10),(self.w-5):self.w]=0.5

			self.bc_mask[index] = 0
			self.bc_mask[index,:,0:3,:]=1
			self.bc_mask[index,:,(self.h-3):self.h,:]=1
			self.bc_mask[index,:,:,0:5]=1
			self.bc_mask[index,:,:,(self.w-5):self.w]=1
			self.bc_mask[index,:,(object_y-object_h):(object_y+object_h),(object_x-object_w):(object_x+object_w)] = 1
			self.env_info[index]["type"] = type
		
			
	
	def update_env(self,index):
		"""
		update boundary conditions of environments[indices]
		"""
		# update state
		self.a_old[index] = self.a_new[index]
		
		# update boundary conditions
		if self.env_info[index]["type"] == "magnus":
			object_r = self.env_info[index]["r"]
			vx_old = self.env_info[index]["vx"]
			vy_old = self.env_info[index]["vy"]
			
			if not self.interactive:
				flow_v = self.env_info[index]["flow_v"]
				object_w = self.env_info[index]["w"]
				object_vx = vx_old*self.brown_damping + self.brown_velocity*np.random.randn()
				object_vy = vy_old*self.brown_damping + self.brown_velocity*np.random.randn()
				
				object_x = self.env_info[index]["x"]+(vx_old+object_vx)/2*self.dt
				object_y = self.env_info[index]["y"]+(vy_old+object_vy)/2*self.dt
				
				if object_x < object_r + 10:
					object_x = object_r + 10
					object_vx = -object_vx
				if object_x > self.w - object_r - 10:
					object_x = self.w - object_r - 10
					object_vx = -object_vx
					
				if object_y < object_r + 10:
					object_y = object_r+10
					object_vy = -object_vy
				if object_y > self.h - object_r - 10:
					object_y = self.h - object_r - 10
					object_vy = -object_vy
				
			if self.interactive:
				flow_v = self.mousev
				object_w = self.mousew/object_r
				object_vx = max(min((self.mousex-self.env_info[index]["x"])/self.interactive_spring,self.max_speed),-self.max_speed)
				object_vy = max(min((self.mousey-self.env_info[index]["y"])/self.interactive_spring,self.max_speed),-self.max_speed)
				
				object_x = self.env_info[index]["x"]+(vx_old+object_vx)/2*self.dt
				object_y = self.env_info[index]["y"]+(vy_old+object_vy)/2*self.dt
				
				if object_x < object_r + 10:
					object_x = object_r + 10
					object_vx = 0
				if object_x > self.w - object_r - 10:
					object_x = self.w - object_r - 10
					object_vx = 0
					
				if object_y < object_r + 10:
					object_y = object_r+10
					object_vy = 0
				if object_y > self.h - object_r - 10:
					object_y = self.h - object_r - 10
					object_vy = 0
			
			
			# 1. generate mesh 2 x [2r x 2r]
			y_mesh,x_mesh = toCuda(torch.meshgrid([torch.arange(-object_r,object_r+1),torch.arange(-object_r,object_r+1)]))
			
			# 2. generate mask
			mask_ball = ((x_mesh**2+y_mesh**2)<object_r**2).float().unsqueeze(0)
			
			# 3. generate bc_values and multiply with mask
			v_ball = object_w*torch.cat([x_mesh.unsqueeze(0),-y_mesh.unsqueeze(0)])*mask_ball
			
			# 4. add masks / bc_values
			self.bc_mask[index,:,:,:]=0
			self.bc_mask[index,:,0:3,:]=1
			self.bc_mask[index,:,(self.h-3):self.h,:]=1
			self.bc_mask[index,:,:,0:5]=1
			self.bc_mask[index,:,:,(self.w-5):self.w]=1
			
			self.bc_mask[index,:,int(object_y-object_r):int(object_y+object_r+1),int(object_x-object_r):int(object_x+object_r+1)] += mask_ball
			self.bc_values[index,0,int(object_y-object_r):int(object_y+object_r+1),int(object_x-object_r):int(object_x+object_r+1)] = v_ball[0]+object_vy
			self.bc_values[index,1,int(object_y-object_r):int(object_y+object_r+1),int(object_x-object_r):int(object_x+object_r+1)] = v_ball[1]+object_vx
			self.bc_values[index] = self.bc_values[index]*self.bc_mask[index]
			
			self.bc_values[index,1,10:(self.h-10),0:5]=flow_v
			self.bc_values[index,1,10:(self.h-10),(self.w-5):self.w]=flow_v
			
			self.env_info[index]["x"] = object_x
			self.env_info[index]["y"] = object_y
			self.env_info[index]["vx"] = object_vx
			self.env_info[index]["vy"] = object_vy
			self.env_info[index]["flow_v"] = flow_v
		
		if self.env_info[index]["type"] == "DFG_benchmark":
			object_r = self.env_info[index]["r"]
			vx_old = self.env_info[index]["vx"]
			vy_old = self.env_info[index]["vy"]
			
			if not self.interactive:
				flow_v = self.env_info[index]["flow_v"]
				object_w = self.env_info[index]["w"]
				object_vx = vx_old*self.brown_damping + self.brown_velocity*np.random.randn()
				object_vy = vy_old*self.brown_damping + self.brown_velocity*np.random.randn()
				
				object_x = self.env_info[index]["x"]+(vx_old+object_vx)/2*self.dt
				object_y = self.env_info[index]["y"]+(vy_old+object_vy)/2*self.dt
				
				if object_x < object_r + self.padding_x + 1:
					object_x = object_r + self.padding_x + 1
					object_vx = -object_vx
				if object_x > self.w - object_r - self.padding_x - 1:
					object_x = self.w - object_r - self.padding_x - 1
					object_vx = -object_vx
					
				if object_y < object_r + self.padding_y + 1:
					object_y = object_r + self.padding_y + 1
					object_vy = -object_vy
				if object_y > self.h - object_r - self.padding_y - 1:
					object_y = self.h - object_r - self.padding_y - 1
					object_vy = -object_vy
				
			if self.interactive:
				flow_v = self.mousev
				object_w = self.mousew/object_r
				object_vx = max(min((self.mousex-self.env_info[index]["x"])/self.interactive_spring,self.max_speed),-self.max_speed)
				object_vy = max(min((self.mousey-self.env_info[index]["y"])/self.interactive_spring,self.max_speed),-self.max_speed)
				
				object_x = self.env_info[index]["x"]+(vx_old+object_vx)/2*self.dt
				object_y = self.env_info[index]["y"]+(vy_old+object_vy)/2*self.dt
				
				if object_x < object_r + self.padding_x + 1:
					object_x = object_r + self.padding_x + 1
					object_vx = 0
				if object_x > self.w - object_r - self.padding_x - 1:
					object_x = self.w - object_r - self.padding_x - 1
					object_vx = 0
					
				if object_y < object_r + self.padding_y + 1:
					object_y = object_r + self.padding_y + 1
					object_vy = 0
				if object_y > self.h - object_r - self.padding_y - 1:
					object_y = self.h - object_r - self.padding_y - 1
					object_vy = 0
			
			self.bc_values[index,:,:,:]=0
			self.bc_mask[index,:,:,:]=0
			self.bc_mask[index,:,0:3,:]=1
			self.bc_mask[index,:,(self.h-3):self.h,:]=1
			self.bc_mask[index,:,:,0:5]=1
			self.bc_mask[index,:,:,(self.w-5):self.w]=1
			
			# 1. generate mesh 2 x [2r x 2r]
			y_mesh,x_mesh = toCuda(torch.meshgrid([torch.arange(-object_r,object_r+1),torch.arange(-object_r,object_r+1)]))
			
			# 2. generate mask
			mask_ball = ((x_mesh**2+y_mesh**2)<object_r**2).float().unsqueeze(0)
			
			# 3. generate bc_values and multiply with mask
			v_ball = object_w*torch.cat([x_mesh.unsqueeze(0),-y_mesh.unsqueeze(0)])*mask_ball
			
			# 4. add masks / bc_values
			x_pos1, y_pos1 = int((object_x-object_r)),int((object_y-object_r))
			x_pos2, y_pos2 = x_pos1+mask_ball.shape[1],y_pos1+mask_ball.shape[2]
			self.bc_mask[index,:,y_pos1:y_pos2,x_pos1:x_pos2] += mask_ball
			self.bc_values[index,0,y_pos1:y_pos2,x_pos1:x_pos2] += v_ball[0]+object_vy
			self.bc_values[index,1,y_pos1:y_pos2,x_pos1:x_pos2] += v_ball[1]+object_vx
			
			# inlet / outlet flow
			profile_size = self.bc_values[index,0,(self.padding_y):-(self.padding_y),:(self.padding_x)].shape[0]
			flow_profile = toCuda(torch.arange(0,profile_size,1.0))
			flow_profile *= 0.41/flow_profile[-1]
			flow_profile = 4*flow_profile*(0.41-flow_profile)/0.1681
			flow_profile = flow_profile.unsqueeze(1)
			self.bc_values[index,1,(self.padding_y):-(self.padding_y),:(self.padding_x)] = flow_v*flow_profile
			self.bc_values[index,1,(self.padding_y):-(self.padding_y),-(self.padding_x):] = flow_v*flow_profile
			
			self.env_info[index]["x"] = object_x
			self.env_info[index]["y"] = object_y
			self.env_info[index]["vx"] = object_vx
			self.env_info[index]["vy"] = object_vy
			self.env_info[index]["w"] = object_w
			self.env_info[index]["flow_v"] = flow_v
		
		if self.env_info[index]["type"] == "box":
			object_h = self.env_info[index]["h"]
			object_w = self.env_info[index]["w"]
			vx_old = self.env_info[index]["vx"]
			vy_old = self.env_info[index]["vy"]
			
			if not self.interactive:
				flow_v = self.env_info[index]["flow_v"]
				object_vx = vx_old*self.brown_damping + self.brown_velocity*np.random.randn()
				object_vy = vy_old*self.brown_damping + self.brown_velocity*np.random.randn()
				
				object_x = self.env_info[index]["x"]+(vx_old+object_vx)/2*self.dt
				object_y = self.env_info[index]["y"]+(vy_old+object_vy)/2*self.dt
				
				if object_x < object_w + 10:
					object_x = object_w + 10
					object_vx = -object_vx
				if object_x > self.w - object_w - 10:
					object_x = self.w - object_w - 10
					object_vx = -object_vx
					
				if object_y < object_h + 10:
					object_y = object_h+10
					object_vy = -object_vy
				if object_y > self.h - object_h - 10:
					object_y = self.h - object_h - 10
					object_vy = -object_vy
				
			if self.interactive:
				flow_v = self.mousev
				object_vx = max(min((self.mousex-self.env_info[index]["x"])/self.interactive_spring,self.max_speed),-self.max_speed)
				object_vy = max(min((self.mousey-self.env_info[index]["y"])/self.interactive_spring,self.max_speed),-self.max_speed)
				
				object_x = self.env_info[index]["x"]+(vx_old+object_vx)/2*self.dt
				object_y = self.env_info[index]["y"]+(vy_old+object_vy)/2*self.dt
				
				if object_x < object_w + 10:
					object_x = object_w + 10
					object_vx = 0
				if object_x > self.w - object_w - 10:
					object_x = self.w - object_w - 10
					object_vx = 0
					
				if object_y < object_h + 10:
					object_y = object_h+10
					object_vy = 0
				if object_y > self.h - object_h - 10:
					object_y = self.h - object_h - 10
					object_vy = 0
			
			
			self.bc_mask[index,:,:,:]=0
			self.bc_mask[index,:,0:3,:]=1
			self.bc_mask[index,:,(self.h-3):self.h,:]=1
			self.bc_mask[index,:,:,0:5]=1
			self.bc_mask[index,:,:,(self.w-5):self.w]=1
			
			self.bc_mask[index,:,int(object_y-object_h):int(object_y+object_h),int(object_x-object_w):int(object_x+object_w)] = 1
			self.bc_values[index,0,int(object_y-object_h):int(object_y+object_h),int(object_x-object_w):int(object_x+object_w)] = object_vy
			self.bc_values[index,1,int(object_y-object_h):int(object_y+object_h),int(object_x-object_w):int(object_x+object_w)] = object_vx
			
			self.bc_values[index] = self.bc_values[index]*self.bc_mask[index]
			self.bc_values[index,1,10:(self.h-10),0:5]=flow_v
			self.bc_values[index,1,10:(self.h-10),(self.w-5):self.w]=flow_v
			
			self.env_info[index]["x"] = object_x
			self.env_info[index]["y"] = object_y
			self.env_info[index]["vx"] = object_vx
			self.env_info[index]["vy"] = object_vy
			self.env_info[index]["flow_v"] = flow_v
			
		if self.env_info[index]["type"] == "pipe":
			if not self.interactive:
				flow_v = self.env_info[index]["flow_v"]
			if self.interactive:
				flow_v = self.mousev
				self.bc_values[index] = self.bc_values[index]/self.env_info[index]["flow_v"]*flow_v
			self.env_info[index]["flow_v"] = flow_v
			
		if self.env_info[index]["type"] == "image":
			object_h = self.env_info[index]["h"]
			object_w = self.env_info[index]["w"]
			vx_old = self.env_info[index]["vx"]
			vy_old = self.env_info[index]["vy"]
			
			image_mask = images[self.env_info[index]["image"]]
			background_image_mask = backgrounds[self.env_info[index]["background_image"]]
			
			if not self.interactive:
				flow_v = self.env_info[index]["flow_v"]
				object_vx = vx_old*self.brown_damping + self.brown_velocity*np.random.randn()
				object_vy = vy_old*self.brown_damping + self.brown_velocity*np.random.randn()
				
				object_x = self.env_info[index]["x"]+(vx_old+object_vx)/2*self.dt
				object_y = self.env_info[index]["y"]+(vy_old+object_vy)/2*self.dt
				
				if object_x < object_w//2 + 10:
					object_x = object_w//2 + 10
					object_vx = -object_vx
				if object_x > self.w - object_w//2 - 10:
					object_x = self.w - object_w//2 - 10
					object_vx = -object_vx
					
				if object_y < object_h//2 + 10:
					object_y = object_h//2+10
					object_vy = -object_vy
				if object_y > self.h - object_h//2 - 10:
					object_y = self.h - object_h//2 - 10
					object_vy = -object_vy
				
			if self.interactive:
				flow_v = self.mousev
				object_vx = max(min((self.mousex-self.env_info[index]["x"])/self.interactive_spring,self.max_speed),-self.max_speed)
				object_vy = max(min((self.mousey-self.env_info[index]["y"])/self.interactive_spring,self.max_speed),-self.max_speed)
				
				object_x = self.env_info[index]["x"]+(vx_old+object_vx)/2*self.dt
				object_y = self.env_info[index]["y"]+(vy_old+object_vy)/2*self.dt
				
				if object_x < object_w//2 + 10:
					object_x = object_w//2 + 10
					object_vx = 0
				if object_x > self.w - object_w//2 - 10:
					object_x = self.w - object_w//2 - 10
					object_vx = 0
					
				if object_y < object_h//2 + 10:
					object_y = object_h//2+10
					object_vy = 0
				if object_y > self.h - object_h//2 - 10:
					object_y = self.h - object_h//2 - 10
					object_vy = 0
			
			
			self.bc_mask[index,:,:,:]=0
			self.bc_mask[index,:,0:3,:]=1
			self.bc_mask[index,:,(self.h-3):self.h,:]=1
			self.bc_mask[index,:,:,0:5]=1
			self.bc_mask[index,:,:,(self.w-5):self.w]=1
			
			
			self.bc_mask[index,:] = 1-(1-self.bc_mask[index,:])*(1-background_image_mask)
			self.bc_mask[index,:,int(object_y-object_h//2):int(object_y-object_h//2+object_h),int(object_x-object_w//2):int(object_x-object_w//2+object_w)] = 1-(1-self.bc_mask[index,:,int(object_y-object_h//2):int(object_y-object_h//2+object_h),int(object_x-object_w//2):int(object_x-object_w//2+object_w)])*(1-image_mask)
			
			
			self.bc_values[index,:]=0
			self.bc_values[index,0,int(object_y-object_h//2):int(object_y-object_h//2+object_h),int(object_x-object_w//2):int(object_x-object_w//2+object_w)] = object_vy*image_mask
			self.bc_values[index,1,int(object_y-object_h//2):int(object_y-object_h//2+object_h),int(object_x-object_w//2):int(object_x-object_w//2+object_w)] = object_vx*image_mask
			
			self.bc_values[index] = self.bc_values[index]*self.bc_mask[index]
			
			self.bc_values[index,1,20:(self.h-20),0:5]=flow_v
			self.bc_values[index,1,20:(self.h-20),(self.w-5):self.w]=flow_v
			
			self.env_info[index]["x"] = object_x
			self.env_info[index]["y"] = object_y
			self.env_info[index]["vx"] = object_vx
			self.env_info[index]["vy"] = object_vy
			self.env_info[index]["flow_v"] = flow_v
			
			
	
	def ask(self):
		"""
		ask for a batch of gradients for fluid simulation
		:return: 
			:gradients: shape: batch_size x 2 x h x w (for vector potential and pressure field)
			:hidden_states: list of hidden_states (length: batch_size; default values: None)
		"""
		self.indices = np.random.choice(self.dataset_size,self.batch_size)
		
		# TODO: compute gradients / loss
		with torch.enable_grad():
			
			# compute gradients wrt a and p
			asked_grads = torch.zeros(self.batch_size,2,self.h,self.w,device=device,requires_grad=True)
			
			#loss(a_old,a_new,p_new,bc_mask,bc_values) # TODO: keep track of old a!
			
			l = loss(self.a_old[self.indices],
				self.a_new[self.indices] + asked_grads[:,0:1],
				self.p[self.indices] + asked_grads[:,1:2],
				self.bc_mask[self.indices],self.bc_values[self.indices],
				mu=self.mus[self.indices],rho=self.rhos[self.indices])
			
			l = l.clamp(min=-20000,max=20000)
			
			l = torch.sum(l) # input grads should be independent of batch size => use sum instead of mean
			
			l.backward()
		
		return asked_grads.grad, [self.hidden_states[i] for i in self.indices]
	
	def tell(self,step,hidden_states=None):
		"""
		:step: update step for accelerations for gradients given by ask
		:hidden_states: list of hidden states that are returned in following ask() calls => this is helpful to store the optimizer state
		:return: loss to optimize neural update-step-model (scalar values)
		"""
		hidden_states = [None for _ in self.indices] if hidden_states is None else hidden_states
		
		self.iterations[self.indices] = self.iterations[self.indices] + 1
		
		# update a_new and p based on step
		
		a_new = self.a_new[self.indices] + step[:,0:1]
		p = self.p[self.indices] + step[:,1:2]
		
		self.a_new[self.indices] = a_new.detach()
		self.p[self.indices] = p.detach()
		
		
		# compute loss
		l = loss(self.a_old[self.indices],
				self.a_new[self.indices] + step[:,0:1],
				self.p[self.indices] + step[:,1:2],
				self.bc_mask[self.indices],self.bc_values[self.indices],
				mu=self.mus[self.indices],rho=self.rhos[self.indices])
		
		l = l.clamp(min=-20000,max=20000)
		
		# update / reset environemts
		if self.interactive:
			self.mousev = min(max(self.mousev,-self.max_speed),self.max_speed)
			self.mousew = min(max(self.mousew,-self.max_speed),self.max_speed)
		
		# update step if iterations_per_timestep is reached
		for i,index in enumerate(self.indices):
			self.hidden_states[index] = hidden_states[i]
			if self.iterations[index] % self.iterations_per_timestep == 0:
				self.T[index] = self.T[index] + self.dt # <- could be moved into update_env...
				self.update_env(index) # TODO: update a_old / a_new
				if l[i] >= 20000 or torch.max(torch.abs(self.a_new[index]))>10000:
					self.reset_env(index)
		
		# reset environments eventually TODO: check that / reset environment, if E_int becomes too large!
		self.step += 1
		if self.step % (self.average_sequence_length*self.iterations_per_timestep/self.batch_size) == 0:#ca x*batch_size steps until env gets reset => TODO attention!: average_sequence_length mut be divisible by (batch_size*iterations_per_timestep)!
			self.reset_env(int(self.reset_i))
			self.reset_i = (self.reset_i+1)%self.dataset_size
		
		return torch.mean(l)

