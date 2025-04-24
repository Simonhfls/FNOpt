import torch
import numpy as np
from get_param2 import params,toCuda,toCpu,device
from utils import log_range_params, range_params
from derivatives import laplace, laplace_detach, dx, dy
from derivatives import laplace_periodic, laplace_periodic_detach, dx_periodic, dy_periodic

eps = 1e-7

"""
dataset for Burgers equation
ask-tell interface:
ask(): ask for batch of gradients of velocities wrt certain loss function
tell(): tell update step for velocities (positions are updated internally) => return loss to update NN parameters
"""
#Attention: x/y are swapped (x-dimension=1; y-dimension=0)


class InitPeriodicCond2d(): # taken from: https://github.com/cics-nd/ar-pde-cnn/blob/master/2D-Burgers-SWAG/utils/burgerLoader2D.py
	"""Generate periodic initial condition on the fly.

	Args:
		order (int): order of Fourier series expansion
		ncells (int): spatial discretization over [0, 1]
		nsamples (int): total # samples
	"""
	def __init__(self, ncells, nsamples, order=4):
		super().__init__()
		self.order = order
		self.nsamples = nsamples
		self.ncells = ncells
		x = np.linspace(0, 1, ncells+1)[:-1]
		xx, yy = np.meshgrid(x, x)
		aa, bb = np.meshgrid(np.arange(-order, order+1), np.arange(-order, order+1))
		k = np.stack((aa.flatten(), bb.flatten()), 1)
		self.kx_plus_ly = (np.outer(k[:, 0], xx.flatten()) + np.outer(k[:, 1], yy.flatten()))*2*np.pi

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index
		Returns:
			init_condition
		"""
		np.random.seed(index+100000) # Make sure this is different than the seeds set in finite element solver!
		lam = np.random.randn(2, 2, (2*self.order+1)**2)
		c = -1 + np.random.rand(2) * 2

		f = np.dot(lam[0], np.cos(self.kx_plus_ly)) + np.dot(lam[1], np.sin(self.kx_plus_ly))
		f = 2 * f / np.amax(np.abs(f), axis=1, keepdims=True) + c[:, None]
		return torch.FloatTensor(f).reshape(-1, self.ncells, self.ncells)

	def __len__(self):
		# generate on-the-fly
		return self.nsamples

ic = InitPeriodicCond2d(params.data.height,1)

dt = params.cloth.dt

def loss(v_old,v_new,mu,bc_mask,bc_values):
	"""
	:v_old: old velocity field (shape: batch_size x 2 x h x w)
	:v_new: new velocity field (shape: batch_size x 2 x h x w)
	:mu: viscosity (shape: batch_size x 1 x 1 x 1)
	:bc_mask: mask for dirichlet boundary conditions (shape: batch_size x 1 x h x w)
	:bc_values: values for dirichlet boundary conditions (shape: batch_size x 1 x h x w)
	"""
	v_new = bc_mask * bc_values + (1-bc_mask)*v_new
	
	residuals_u = (v_new[:,0:1]-v_old[:,0:1])/dt + v_new[:,0:1]*dx_periodic(v_new[:,0:1]) + v_new[:,1:2]*dy_periodic(v_new[:,0:1]) - mu*laplace_periodic_detach(v_new[:,0:1])
	residuals_v = (v_new[:,1:2]-v_old[:,1:2])/dt + v_new[:,0:1]*dx_periodic(v_new[:,1:2]) + v_new[:,1:2]*dy_periodic(v_new[:,1:2]) - mu*laplace_periodic_detach(v_new[:,1:2])
	#residuals_u = (v_new[:,0:1]-v_old[:,0:1])/dt + v_new[:,0:1]*dx(v_new[:,0:1].detach()) + v_new[:,1:2]*dy(v_new[:,0:1].detach()) - mu*laplace_detach(v_new[:,0:1])
	#residuals_v = (v_new[:,1:2]-v_old[:,1:2])/dt + v_new[:,0:1]*dx(v_new[:,1:2].detach()) + v_new[:,1:2]*dy(v_new[:,1:2].detach()) - mu*laplace_detach(v_new[:,1:2])
	
	#residuals_u = (v_new[:,0:1]-v_old[:,0:1])/dt + v_new[:,0:1]*dx(v_new[:,0:1]) + v_new[:,1:2]*dy(v_new[:,0:1]) - mu*laplace_detach(v_new[:,0:1])
	#residuals_v = (v_new[:,1:2]-v_old[:,1:2])/dt + v_new[:,0:1]*dx(v_new[:,1:2]) + v_new[:,1:2]*dy(v_new[:,1:2]) - mu*laplace_detach(v_new[:,1:2])
	
	return (torch.sum((1-bc_mask)*residuals_u**2,dim=[1,2,3]) + torch.sum((1-bc_mask)*residuals_v**2,dim=[1,2,3]))/torch.sum((1-bc_mask))

class DatasetBurgers:
	def __init__(self,h,w,batch_size=100,dataset_size=1000,average_sequence_length=5000,iterations_per_timestep=5,mu_range=None):
		
		# dataset parameters
		self.h,self.w = h,w
		self.batch_size = batch_size
		self.dataset_size = dataset_size
		self.average_sequence_length = average_sequence_length
		mu_range = [params.Fluid.min_mu,params.Fluid.mu] if mu_range is None else mu_range
		self.mu_range = log_range_params(mu_range)
		
		# grid utility
		x_space = torch.linspace(0,(w-1),w)
		y_space = torch.linspace(-(h-1)/2,(h-1)/2,h)
		y_grid,x_grid = torch.meshgrid(y_space,x_space,indexing="ij")
		self.y_mesh,self.x_mesh = toCuda(torch.meshgrid([torch.arange(0,self.h),torch.arange(0,self.w)]))
		
		# concentration state values
		self.v_old = toCuda(torch.zeros(self.dataset_size,2,h,w)) # old concentration
		self.v_new = toCuda(torch.zeros(self.dataset_size,2,h,w)) # new concentration
		#self.mu = toCuda(torch.zeros(self.dataset_size,1,1,1)) # Diffusivity
		self.T = toCuda(torch.zeros(self.dataset_size,1)) # timestep
		self.iterations = toCuda(torch.zeros(dataset_size)) # iterations for the individual training pool samples
		self.iterations_per_timestep = iterations_per_timestep # number of iterations per timestep
		
		# dirichlet boundary conditions
		self.bc_mask = toCuda(torch.zeros(self.dataset_size,1,self.h,self.w)) # solution we are looking for
		self.bc_values = toCuda(torch.zeros(self.dataset_size,2,self.h,self.w)) # solution we are looking for
		
		self.mu = toCuda(torch.exp(self.mu_range[0]+torch.rand(self.dataset_size,1,1,1)*self.mu_range[1])) # Diffusivity
		
		self.hidden_states = [None for _ in range(dataset_size)]
		
		for i in range(self.dataset_size):
			self.reset_env(i)
		
		self.step = 0 # number of tell()-calls
		self.reset_i = 0
		
	def reset_env(self,index):
		
		#print(f"reset diffusion [{index}] (D={self.D[index]}) after {self.iterations[index]} iterations")
		
		# reset state
		self.hidden_states[index] = None # hidden state for (neural) optimizer
		self.T[index] = 0
		self.v_new[index] = ic[np.random.randint(0,10000000)]#0
		self.v_old[index] = self.v_new[index] # TODO: add different initial conditions
		self.iterations[index] = 0
		
		# new environment
		trivial_setup = 0 if torch.rand(1)<0.2 else 1 # trivial setup (no Diffusivity / velocity) in 20% of cases
		#self.mu[index] = 0.3#trivial_setup*torch.exp(self.mu_range[0]+torch.rand(1)*self.mu_range[1])
		self.T[index] = 0
		
		# setup dirichlet boundary conditions
		self.bc_mask[index] = 1
		self.bc_mask[index,:,2:-2,2:-2] = 0 # codo: add further areas for bc
		"""
		for _ in range(3):
			x_pos = torch.randint(10,self.w-10,[1])
			y_pos = torch.randint(10,self.h-10,[1])
			w,h = torch.randint(0,20,[2])
			self.bc_mask[index,:,y_pos:(y_pos+h),x_pos:(x_pos+w)] = 1
		"""
		for i in range(2):
			phase = torch.rand(1,1,device=device)*2*3.14
			direction = torch.randn(1,1,device=device)*self.x_mesh+torch.randn(1,1,device=device)*self.y_mesh
			self.bc_values[index,i,:,:] = 0#3*torch.sin(phase+direction*0.1) # CODO: add further noise...
		
		self.v_new[index] = self.bc_mask[index] * self.bc_values[index] + (1-self.bc_mask[index])*self.v_new[index]
		self.v_old[index] = self.bc_mask[index] * self.bc_values[index] + (1-self.bc_mask[index])*self.v_old[index]
	
	def reset0_env(self,index):
		# reset state
		self.hidden_states[index] = None # hidden state for (neural) optimizer
		self.T[index] = 0
		#self.v_new[index] = ic[7]#ic[np.random.randint(0,10000000)]#0
		self.v_new[index] = ic[np.random.randint(0,10000000)]#0
		self.v_old[index] = self.v_new[index] # TODO: add different initial conditions
		self.iterations[index] = 0
		
		# new environment
		self.mu[index] = 1#0.3
		self.T[index] = 0
		
		# setup dirichlet boundary conditions
		self.bc_mask[index] = 1
		self.bc_mask[index,:,2:-2,2:-2] = 0 # codo: add further areas for bc
		
		self.v_new[index] = self.bc_mask[index] * self.bc_values[index] + (1-self.bc_mask[index])*self.v_new[index]
		self.v_old[index] = self.bc_mask[index] * self.bc_values[index] + (1-self.bc_mask[index])*self.v_old[index]
	
	def update_env(self,index):
		
		# update state
		self.v_old[index] = self.v_new[index]
		
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
			asked_grads = torch.zeros(self.batch_size,2,self.h,self.w,device=device,requires_grad=True)
			
			l = loss(self.v_old[self.indices],self.v_new[self.indices]+asked_grads,self.mu[self.indices],self.bc_mask[self.indices],self.bc_values[self.indices])
			
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
		
		v_new = self.v_new[self.indices] + step
		self.v_new[self.indices] = v_new.detach()
		self.v_new[self.indices] = self.bc_mask[self.indices] * self.bc_values[self.indices] + (1-self.bc_mask[self.indices])*self.v_new[self.indices]
		
		l = loss(self.v_old[self.indices],v_new,self.mu[self.indices],self.bc_mask[self.indices],self.bc_values[self.indices])
		
		
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
