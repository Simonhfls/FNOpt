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

dt = params.dt

def loss(x_old,v_old,a,c,bc_mask,bc_values):
	"""
	:x_old: old height field (shape: batch_size x 1 x h x w)
	:v_old: old velocity field (shape: batch_size x 1 x h x w)
	:a: accelerations (shape: batch_size x 2 x h x w)
	:c: "spring constant" / wave speed (shape: batch_size x 1 x h x w)
	:bc_mask: mask for dirichlet boundary conditions (shape: batch_size x 1 x h x w)
	:bc_values: values for dirichlet boundary conditions (shape: batch_size x 1 x h x w)
	"""
	
	v_new = v_old + dt*a
	x_new = x_old + dt*v_new
	
	# boundary conditions:
	v_new = (1-bc_mask)*v_new # velocity = 0 at boundaries
	x_new = bc_mask * bc_values + (1-bc_mask)*x_new
	
	# attention: we have to use (v_new-v_old)/dt instead of directly using a in order to avoid misleading gradients at the boundaries!
	#residuals = (v_new-v_old)/dt - c**2*laplace(x_old) # this would result in simple explicit forward euler (toy example that makes not really sense)
	#residuals = (v_new-v_old)/dt - c**2*laplace(x_new) # implicit scheme leads to high numerical dissipation
	#residuals = (v_new-v_old)/dt - c**2*laplace(0.5*(x_new+x_old)) # "imex" scheme => centerd finite differences in time
	residuals = (v_new-v_old)/dt - c**2*laplace_detach(0.5*(x_new+x_old)) # "imex" scheme => centerd finite differences in time
	return torch.sum((1-bc_mask)*residuals**2,dim=[1,2,3])/torch.sum(1-bc_mask)

class DatasetWave:
	def __init__(self,h,w,batch_size=100,dataset_size=1000,average_sequence_length=5000,iterations_per_timestep=5,c_range=None):
		
		# dataset parameters
		self.h,self.w = h,w
		self.batch_size = batch_size
		self.dataset_size = dataset_size
		self.average_sequence_length = average_sequence_length
		c_range = [params.min_c,params.c] if c_range is None else c_range
		self.c_range = log_range_params(c_range)
		# TODO: frequency range
		
		# grid utility
		x_space = torch.linspace(0,(w-1),w)
		y_space = torch.linspace(-(h-1)/2,(h-1)/2,h)
		y_grid,x_grid = torch.meshgrid(y_space,x_space,indexing="ij")
		self.y_mesh,self.x_mesh = toCuda(torch.meshgrid([torch.arange(0,self.h),torch.arange(0,self.w)]))
		
		# concentration state values
		self.x_old = toCuda(torch.zeros(self.dataset_size,1,h,w)) # old height field
		self.v_old = toCuda(torch.zeros(self.dataset_size,1,h,w)) # old velocity field
		self.a = toCuda(torch.zeros(self.dataset_size,1,h,w)) # accelerations
		#self.c = toCuda(torch.zeros(self.dataset_size,1,1,1)) # wave speed
		self.T = toCuda(torch.zeros(self.dataset_size,1)) # timestep
		self.iterations = toCuda(torch.zeros(dataset_size)) # iterations for the individual training pool samples
		self.iterations_per_timestep = iterations_per_timestep # number of iterations per timestep
		self.c = toCuda(torch.exp(self.c_range[0]+torch.rand(self.dataset_size,1,1,1)*self.c_range[1])) # wave speed
		
		# dirichlet boundary conditions
		self.bc_mask = toCuda(torch.zeros(self.dataset_size,1,self.h,self.w)) # solution we are looking for
		self.bc_values = toCuda(torch.zeros(self.dataset_size,1,self.h,self.w)) # solution we are looking for
		
		# values to make setting the boundary condition values easier
		self.bc_magnitudes = toCuda(torch.zeros(self.dataset_size,1,self.h,self.w))
		self.bc_frequencies = toCuda(torch.zeros(self.dataset_size,1,self.h,self.w))
		self.bc_phases = toCuda(torch.zeros(self.dataset_size,1,self.h,self.w))
		
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
		self.x_old[index] = 0
		self.v_old[index] = 0
		self.a[index] = 0
		self.iterations[index] = 0
		
		# new environment
		#self.c[index] = torch.exp(self.c_range[0]+torch.rand(1)*self.c_range[1])
		self.T[index] = 0
		
		# setup dirichlet boundary conditions
		self.bc_mask[index] = 1
		self.bc_mask[index,:,2:-2,2:-2] = 0 # codo: add further areas for bc
		
		for x in [-20,0,20]:
			self.bc_mask[index,:,(self.h//2-2):(self.h//2+2),(self.w//2+x-2):(self.w//2+x+2)] = 1
			
		self.bc_magnitudes[index] = 0
		self.bc_magnitudes[index,0,2:-2,2:-2] = 1
		self.bc_frequencies[index] = 0.5#1#
		self.bc_phases[index] = 0
		
		self.bc_values[index] = self.bc_magnitudes[index]*torch.sin(self.bc_phases[index]) # CODO: add further noise...
		
		self.x_old[index] = self.bc_mask[index] * self.bc_values[index] + (1-self.bc_mask[index])*self.x_old[index]
		self.v_old[index] = (1-self.bc_mask[index])*self.v_old[index]
	
	def update_env(self,index):
		
		#print(f"c: {self.c[index]} f: {self.bc_frequencies[index]} phase: {self.bc_phases[index]+self.T[index]*self.bc_frequencies[index]}")
		# update state
		#self.v_old[index] = self.v_old[index] + dt*self.c[index]**2*laplace(self.x_old[index]) # <- simple explicit euler
		self.v_old[index] = self.v_old[index] + dt*self.a[index]
		self.x_old[index] = self.x_old[index] + dt*self.v_old[index]
		self.a[index] = 0
		
		# update boundary conditions
		self.bc_values[index] = self.bc_magnitudes[index]*torch.sin(self.bc_phases[index]+self.T[index]*self.bc_frequencies[index]) # CODO: add further noise...
		
		# CODO: update bc_mask...
		
		self.x_old[index] = self.bc_mask[index] * self.bc_values[index] + (1-self.bc_mask[index])*self.x_old[index]
		self.v_old[index] = (1-self.bc_mask[index])*self.v_old[index]
	
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
			
			l = loss(self.x_old[self.indices],self.v_old[self.indices],self.a[self.indices]+asked_grads,self.c[self.indices],self.bc_mask[self.indices],self.bc_values[self.indices])
			
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
		
		a_new = self.a[self.indices] + step
		self.a[self.indices] = a_new.detach()
		
		l = loss(self.x_old[self.indices],self.v_old[self.indices],a_new,self.c[self.indices],self.bc_mask[self.indices],self.bc_values[self.indices])
		
		# update step if iterations_per_timestep is reached
		for i,index in enumerate(self.indices):
			self.hidden_states[index] = hidden_states[i]
			if self.iterations[index] % self.iterations_per_timestep == 0:
				self.T[index] = self.T[index] + dt
				self.update_env(index)
				#if l[i] > 10000:
				#	self.reset_env(index)
		
		# reset environments eventually
		self.step += 1
		if self.step % (self.average_sequence_length*self.iterations_per_timestep/self.batch_size) == 0:#ca x*batch_size steps until env gets reset => TODO attention!: average_sequence_length mut be divisible by (batch_size*iterations_per_timestep)!
			self.reset_env(int(self.reset_i))
			self.reset_i = (self.reset_i+1)%self.dataset_size
			
		return torch.mean(l)
