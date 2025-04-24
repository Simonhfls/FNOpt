import torch
import numpy as np
from get_param2 import params,toCuda,toCpu,device
from utils import log_range_params, range_params
from derivatives import laplace, laplace_detach, dx, dy

eps = 1e-7

"""
dataset for advection diffusion equation
ask-tell interface:
ask(): ask for batch of gradients of velocities wrt certain loss function
tell(): tell update step for velocities (positions are updated internally) => return loss to update NN parameters
"""
#Attention: x/y are swapped (x-dimension=1; y-dimension=0)

# CODO: spatially varying stiffness / shearing / bending parameters

dt = params.cloth.dt

def loss(c_old,c_new,v,R,D,bc_mask,bc_values):
	"""
	:c_old: old concentration (shape: batch_size x 1 x h x w)
	:c_new: new concentration (shape: batch_size x 1 x h x w)
	:v: velocity field for advection term  (shape: batch_size x 2 x h x w)
	:R: source or sinks  (shape: batch_size x 1 x h x w)
	:D: Diffusivity
	:bc_mask: mask for dirichlet boundary conditions (shape: batch_size x 1 x h x w)
	:bc_values: values for dirichlet boundary conditions (shape: batch_size x 1 x h x w)
	"""
	c_new = bc_mask * bc_values + (1-bc_mask)*c_new
	
	advection_term = v[:,1:2]*dx(c_new)+v[:,0:1]*dy(c_new) + (dx(v[:,1:2])+dy(v[:,0:1]))*c_new
	#advection_term = v[:,1:2]*dx(c_old)+v[:,0:1]*dy(c_old) + (dx(v[:,1:2])+dy(v[:,0:1]))*c_new
	
	residuals = (c_new-c_old)/dt - D*laplace(c_new) + advection_term - R # TODO: check if correct
	return torch.sum((1-bc_mask)*residuals**2,dim=[1,2,3])/torch.sum(1-bc_mask)

def loss(c_old,c_new,v,R,D,bc_mask,bc_values): # loss with detached neighborhood gradients
	"""
	:c_old: old concentration (shape: batch_size x 1 x h x w)
	:c_new: new concentration (shape: batch_size x 1 x h x w)
	:v: velocity field for advection term  (shape: batch_size x 2 x h x w)
	:R: source or sinks  (shape: batch_size x 1 x h x w)
	:D: Diffusivity
	:bc_mask: mask for dirichlet boundary conditions (shape: batch_size x 1 x h x w)
	:bc_values: values for dirichlet boundary conditions (shape: batch_size x 1 x h x w)
	"""
	c_new = bc_mask * bc_values + (1-bc_mask)*c_new
	
	#advection_term = v[:,1:2]*dx(c_new)+v[:,0:1]*dy(c_new) + (dx(v[:,1:2])+dy(v[:,0:1]))*c_new
	advection_term = v[:,1:2]*dx(c_new.detach())+v[:,0:1]*dy(c_new.detach()) + (dx(v[:,1:2])+dy(v[:,0:1]))*c_new # => kann das zu instabilität führen?
	#advection_term = v[:,1:2]*dx(c_old)+v[:,0:1]*dy(c_old) + (dx(v[:,1:2])+dy(v[:,0:1]))*c_new # "explicit" forward version (not as stable)
	
	residuals = (c_new-c_old)/dt - D*laplace_detach(c_new) + advection_term - R # TODO: check if correct
	#return torch.sum(residuals**2,dim=[1,2,3])
	return torch.sum((1-bc_mask)*residuals**2,dim=[1,2,3])/torch.sum(1-bc_mask)

class DatasetDiffusion:
	def __init__(self,h,w,batch_size=100,dataset_size=1000,average_sequence_length=5000,iterations_per_timestep=5,D_range=None):
		
		# dataset parameters
		self.h,self.w = h,w
		self.batch_size = batch_size
		self.dataset_size = dataset_size
		self.average_sequence_length = average_sequence_length
		self.D_range = log_range_params(D_range)
		
		# grid utility
		x_space = torch.linspace(0,(w-1),w)
		y_space = torch.linspace(-(h-1)/2,(h-1)/2,h)
		y_grid,x_grid = torch.meshgrid(y_space,x_space,indexing="ij")
		self.y_mesh,self.x_mesh = toCuda(torch.meshgrid([torch.arange(0,self.h),torch.arange(0,self.w)]))
		
		# concentration state values
		self.c_old = toCuda(torch.zeros(self.dataset_size,1,h,w)) # old concentration
		self.c_new = toCuda(torch.zeros(self.dataset_size,1,h,w)) # new concentration
		self.v = toCuda(torch.zeros(self.dataset_size,2,h,w)) # advection velocity
		self.R = toCuda(torch.zeros(self.dataset_size,1,h,w)) # source / sink terms
		#self.D = toCuda(torch.zeros(self.dataset_size,1,1,1)) # Diffusivity
		self.T = toCuda(torch.zeros(self.dataset_size,1)) # timestep
		self.iterations = toCuda(torch.zeros(dataset_size)) # iterations for the individual training pool samples
		self.iterations_per_timestep = iterations_per_timestep # number of iterations per timestep
		
		# dirichlet boundary conditions
		self.bc_mask = toCuda(torch.zeros(self.dataset_size,1,self.h,self.w)) # solution we are looking for
		self.bc_values = toCuda(torch.zeros(self.dataset_size,1,self.h,self.w)) # solution we are looking for
		
		self.hidden_states = [None for _ in range(dataset_size)]
		
		self.D = toCuda(torch.exp(self.D_range[0]+torch.rand(self.dataset_size,1,1,1)*self.D_range[1])*(torch.rand(self.dataset_size,1,1,1)<0.9)) # Diffusivity
		
		
		for i in range(self.dataset_size):
			self.reset_env(i)
		
		self.step = 0 # number of tell()-calls
		self.reset_i = 0
		
	def reset_env(self,index):
		
		#print(f"reset diffusion [{index}] (D={self.D[index]}) after {self.iterations[index]} iterations")
		
		# reset state
		self.hidden_states[index] = None # hidden state for (neural) optimizer
		self.T[index] = 0
		self.c_old[index] = 0
		self.c_new[index] = 0
		self.iterations[index] = 0
		
		# new environment
		trivial_setup = 0 if torch.rand(1)<0. else 1 # trivial setup (no Diffusivity / velocity) in 20% of cases
		self.v[index] = 0
		self.v[index,:] = trivial_setup*torch.randn(2,1,1)*5 # TODO: add more variability
		self.R[index] = 0
		#self.D[index] = trivial_setup*torch.exp(self.D_range[0]+torch.rand(1)*self.D_range[1])
		self.T[index] = 0
		
		# setup dirichlet boundary conditions
		self.bc_mask[index] = 1
		self.bc_mask[index,:,2:-2,2:-2] = 0 # codo: add further areas for bc
		
		for _ in range(3):
			x_pos = torch.randint(10,self.w-10,[1])
			y_pos = torch.randint(10,self.h-10,[1])
			w,h = torch.randint(0,20,[2])
			self.bc_mask[index,:,y_pos:(y_pos+h),x_pos:(x_pos+w)] = 1
		
		phase = torch.rand(1,1,device=device)*2*3.14
		direction = torch.randn(1,1,device=device)*self.x_mesh+torch.randn(1,1,device=device)*self.y_mesh
		self.bc_values[index,0,:,:] = torch.sin(phase+direction*0.1) # CODO: add further noise...
		
		self.c_new[index] = self.bc_mask[index] * self.bc_values[index] + (1-self.bc_mask[index])*self.c_new[index]
		self.c_old[index] = self.bc_mask[index] * self.bc_values[index] + (1-self.bc_mask[index])*self.c_old[index]
	
	def reset0_env(self,index):
		# reset state
		self.hidden_states[index] = None # hidden state for (neural) optimizer
		self.T[index] = 0
		self.c_old[index] = 0
		self.c_new[index] = 0
		
		y = 2*(self.x_mesh/self.h-0.5)
		x = 2*(self.y_mesh/self.w-0.5)
		
		# new environment
		self.v[index] = 0
		self.v[index,0] = -1.5*y#5#0#2#torch.randn(2,1,1)*5
		self.v[index,1] = 1.5*x#5#0#2#torch.randn(2,1,1)*5
		self.R[index] = 0
		self.D[index] = 0.3#torch.exp(self.D_range[0]+torch.rand(1)*self.D_range[1])
		self.T[index] = 0
		
		# setup dirichlet boundary conditions
		self.bc_mask[index] = 1
		self.bc_mask[index,:,2:-2,2:-2] = 0 # codo: add further areas for bc
		#self.bc_mask[index,:,(self.h//2-5):(self.h//2+5),(self.w//2-25):(self.w//2-15)] = 1
		
		#self.R[index,:,(self.h//2-5):(self.h//2+5),(self.w//2-25):(self.w//2-15)] = 1
		self.R[index,0] = torch.exp(-(x**2+(y+0.5)**2)/0.002)
		
		self.bc_values[index] = 0
		self.bc_values[index,:,(self.h//2-5):(self.h//2+5),(self.w//2-25):(self.w//2-15)] = 0.5#1
		
		self.c_new[index] = self.bc_mask[index] * self.bc_values[index] + (1-self.bc_mask[index])*self.c_new[index]
		self.c_old[index] = self.bc_mask[index] * self.bc_values[index] + (1-self.bc_mask[index])*self.c_old[index]
	
	
	def update_env(self,index):
		
		# update state
		self.c_old[index] = self.c_new[index]
		
		# CODO update boundary conditions
	
	def ask(self):
		"""
		:return: 
			gradients for new concentrations (shape: batch_size x 1 x h x w)
			hidden_states for optimizer
		"""
		self.indices = np.random.choice(self.dataset_size,self.batch_size)
		
		with torch.enable_grad():
			# compute gradients wrt accelerations
			asked_grads = torch.zeros(self.batch_size,1,self.h,self.w,device=device,requires_grad=True)
			
			l = loss(self.c_old[self.indices],self.c_new[self.indices]+asked_grads,self.v[self.indices],self.R[self.indices],self.D[self.indices],self.bc_mask[self.indices],self.bc_values[self.indices])
			
			l = torch.sum(l) # input grads should be independent of batch size => use sum instead of mean
			
			l.backward()
		
		return asked_grads.grad, [self.hidden_states[i] for i in self.indices]
	
	def tell(self,step, hidden_states=None):
		"""
		:step: update step for accelerations for gradients given by ask
		:hidden_states: list of hidden states that are returned in following ask() calls => this is helpful to store the optimizer state
		:return: loss to optimize neural update-step-model (scalar values)
		"""
		hidden_states = [None for _ in self.indices] if hidden_states is None else hidden_states
		
		self.iterations[self.indices] = self.iterations[self.indices] + 1
		
		c_new = self.c_new[self.indices] + step
		self.c_new[self.indices] = c_new.detach()
		self.c_new[self.indices] = self.bc_mask[self.indices] * self.bc_values[self.indices] + (1-self.bc_mask[self.indices])*self.c_new[self.indices]
		
		l = loss(self.c_old[self.indices],c_new,self.v[self.indices],self.R[self.indices],self.D[self.indices],self.bc_mask[self.indices],self.bc_values[self.indices])
		
		
		# update step if iterations_per_timestep is reached
		for i,index in enumerate(self.indices):
			self.hidden_states[index] = hidden_states[i]
			if self.iterations[index] % self.iterations_per_timestep == 0:
				self.T[index] = self.T[index] + dt
				self.update_env(index)
				if l[i] > 10000:
					self.reset_env(index)
		
		# reset environments eventually
		self.step += 1
		if self.step % (self.average_sequence_length*self.iterations_per_timestep/self.batch_size) == 0:#ca x*batch_size steps until env gets reset => TODO attention!: average_sequence_length mut be divisible by (batch_size*iterations_per_timestep)!
			self.reset_env(int(self.reset_i))
			self.reset_i = (self.reset_i+1)%self.dataset_size
			
		return torch.mean(l)
