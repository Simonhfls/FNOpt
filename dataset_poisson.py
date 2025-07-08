import torch
import numpy as np
from get_param2 import params,toCuda,toCpu,device
import torch.nn.functional as F
from derivatives import laplace,laplace_detach

"""
def loss(u,f,bc_mask,bc_values): # original "naive loss" => leads to very small gradients even if residuals are relatively large
	u = bc_mask * bc_values + (1-bc_mask)*u
	residuals = laplace(u)+f
	return torch.sum(residuals**2,dim=[1,2,3])
	
def loss(u,f,bc_mask,bc_values): # special loss that makes gradients proportional to residuals
	u = bc_mask * bc_values + (1-bc_mask)*u
	residuals = laplace(u)+f
	return torch.sum(((u+residuals).detach()-u)**2,dim=[1,2,3])

"""
def loss(u,f,bc_mask,bc_values): # loss with laplace_detach => works well :)
	u = bc_mask * bc_values + (1-bc_mask)*u
	residuals = (laplace_detach(u)+f)*(1-bc_mask)
	return torch.sum(residuals**2,dim=[1,2,3])/torch.sum(1-bc_mask)
	
"""
def loss(u,f,bc_mask,bc_values):#(self, update):
	# boundary
	u = bc_mask * bc_values + (1-bc_mask)*u
	
	# interior
	laplace_convolution = laplace(u)
	loss_interior = ((laplace_convolution)**2)  * (1. - bc_mask)
	# combined
	loss = torch.sum(loss_interior, dim = [1, 2, 3])

	return loss
"""
	
def loss2(u,f,bc_mask,bc_values): # loss with boundary loss
	l_bound = bc_mask*(u-bc_values)**2
	l_pois = (1-bc_mask)*(laplace(u)+f)**2
	return torch.sum(l_bound+l_pois,dim=[1,2,3]) # TODO: use mean instead
	

class DatasetPoisson:
	# dataset to learn how to solve Poisson PDE: -\Delta u = f

	def __init__(self,h,w,batch_size=100,dataset_size=1000,average_sequence_length=500,tell_loss=True):
		
		# dataset parameters
		self.h,self.w = h,w
		self.batch_size = batch_size
		self.dataset_size = dataset_size
		self.average_sequence_length = average_sequence_length
		
		# grid utility
		x_space = torch.linspace(0,(w-1),w)
		y_space = torch.linspace((h-1)/2,(h-1)/2,h)
		y_grid,x_grid = torch.meshgrid(y_space,x_space,indexing="ij")
		self.y_mesh,self.x_mesh = toCuda(torch.meshgrid([torch.arange(0,self.h),torch.arange(0,self.w)]))
		
		# poisson state values
		self.u = toCuda(torch.zeros(self.dataset_size,1,self.h,self.w)) # solution we are looking for
		
		# f
		self.f = toCuda(torch.zeros(self.dataset_size,1,self.h,self.w)) # solution we are looking for
		
		# dirichlet boundary conditions
		self.bc_mask = toCuda(torch.zeros(self.dataset_size,1,self.h,self.w)) # solution we are looking for
		self.bc_values = toCuda(torch.zeros(self.dataset_size,1,self.h,self.w)) # solution we are looking for
		
		self.hidden_states = [None for _ in range(dataset_size)]
		
		self.iterations = toCuda(torch.zeros(dataset_size)) # iterations for the individual training pool samples
		
		for i in range(self.dataset_size):
			self.reset_env(i)
		
		self.step = 0 # number of tell()-calls
		self.reset_i = 0
		self.tell_loss = tell_loss
		
	def reset_env(self,index):
	
		# reset state
		self.hidden_states[index] = None # hidden state for (neural) optimizer
		self.u[index] = 0
		self.iterations[index] = 0
		
		# update f
		self.f[index] = 0 # TODO: add further data (here, this is =0 for laplace equation))] = 1
		
		phase = torch.rand(1,1,device=device)*2*3.14
		direction = torch.randn(1,1,device=device)*self.x_mesh+torch.randn(1,1,device=device)*self.y_mesh
		#self.f[index,0,:,:] = 0.1*torch.sin(phase+direction*0.1)
		
		# update boundary conditions
		self.bc_mask[index] = 1
		self.bc_mask[index,:,2:-2,2:-2] = 0 # codo: add further areas for bc
		
		for _ in range(3):
			x_pos = torch.randint(10,self.w-10,[1])
			y_pos = torch.randint(10,self.h-10,[1])
			w,h = torch.randint(0,20,[2])
			self.bc_mask[index,:,y_pos:(y_pos+h),x_pos:(x_pos+w)] = 1
		
		phase = torch.rand(1,1,device=device)*2*3.14
		direction = torch.randn(1,1,device=device)*self.x_mesh+torch.randn(1,1,device=device)*self.y_mesh
		self.bc_values[index,0,:,:] = torch.sin(phase+direction*0.1) # TODO: check frequency / add further noise...
		
	def reset0_env(self,index):
	
		# reset state
		self.hidden_states[index] = None # hidden state for (neural) optimizer
		self.u[index] = 0
		self.iterations[index] = 0
		
		# update f
		self.f[index] = 0 # TODO: add further data (here, this is =0 for laplace equation))] = 1
		
		# update boundary conditions
		self.bc_mask[index] = 1
		self.bc_mask[index,:,2:-2,2:-2] = 0 # codo: add further areas for bc
		
		#self.bc_values[index,0,:,:] = (2*self.x_mesh/self.w-1)*(2*self.y_mesh/self.h-1)
		self.bc_values[index,0,:,:] = (self.x_mesh/self.w)*(self.y_mesh/self.h)
		
	
	def reset1_env(self,index): # "curvy field for paper visualization"
	
		# reset state
		self.hidden_states[index] = None # hidden state for (neural) optimizer
		self.u[index] = 0
		self.iterations[index] = 0
		
		# update f
		self.f[index] = 0 # TODO: add further data (here, this is =0 for laplace equation))] = 1
		
		# update boundary conditions
		self.bc_mask[index] = 1
		self.bc_mask[index,:,2:-2,2:-2] = 0 # codo: add further areas for bc
		
		self.bc_values[index,0,:,:] = torch.sin(4*self.x_mesh/self.w-1)*torch.cos(8*self.y_mesh/self.h-1)
		#self.bc_values[index,0,:,:] = (2*self.x_mesh/self.w-1)*(2*self.y_mesh/self.h-1)
		#self.bc_values[index,0,:,:] = (self.x_mesh/self.w)*(self.y_mesh/self.h)
		
	
	def reset2_env(self,index): # "particle charge"
	
		# reset state
		self.hidden_states[index] = None # hidden state for (neural) optimizer
		self.u[index] = 0
		self.iterations[index] = 0
		
		# update f
		self.f[index] = 0 # TODO: add further data (here, this is =0 for laplace equation))] = 1
		#self.f[index,0,(self.h//2-2):(self.h//2+2),(self.w//2-2):(self.w//2+2)] = -0.1 # TODO: add further data (here, this is =0 for laplace equation))] = 1
		self.f[index,0,(self.h//2-2):(self.h//2+2),(self.w//2-12):(self.w//2-8)] = -0.1 # TODO: add further data (here, this is =0 for laplace equation))] = 1
		self.f[index,0,(self.h//2-2):(self.h//2+2),(self.w//2+8):(self.w//2+12)] = 0.1 # TODO: add further data (here, this is =0 for laplace equation))] = 1
		
		# update boundary conditions
		self.bc_mask[index] = 1
		self.bc_mask[index,:,2:-2,2:-2] = 0 # codo: add further areas for bc
		
		self.bc_values[index,0,:,:] = 0#torch.sin(4*self.x_mesh/self.w-1)*torch.cos(8*self.y_mesh/self.h-1)
		#self.bc_values[index,0,:,:] = (2*self.x_mesh/self.w-1)*(2*self.y_mesh/self.h-1)
		#self.bc_values[index,0,:,:] = (self.x_mesh/self.w)*(self.y_mesh/self.h)
		
	
	def ask(self):
		"""
		:return: 
			gradients for accelerations (shape: batch_size x 3 x h x w)
			hidden_states for optimizer
		"""
		self.indices = np.random.choice(self.dataset_size,self.batch_size)
		
		
		# TODO
		with torch.enable_grad():
			# compute gradients wrt accelerations
			asked_grads = torch.zeros(self.batch_size,1,self.h,self.w,device=device,requires_grad=True)
			
			l = loss(self.u[self.indices] + asked_grads,self.f[self.indices],
				self.bc_mask[self.indices],self.bc_values[self.indices])
			
			l = torch.sum(l) # input grads should be independent of batch size => use sum instead of mean
			
			l.backward()
			
		return asked_grads.grad, [self.hidden_states[i] for i in self.indices]
		
	def tell(self,step, hidden_states=None):
		"""
		:step: update step for accelerations for gradients given by ask
		:return: loss to optimize neural update-step-model => TODO: shape sollte batch_size sein, damit loss / gradienten rescaled werden können!
		"""
		hidden_states = [None for _ in self.indices] if hidden_states is None else hidden_states
		
		self.iterations[self.indices] = self.iterations[self.indices] + 1
		
		# TODO
		u = self.u[self.indices] + step
		self.u[self.indices] = u.detach()
		self.u[self.indices] = self.bc_mask[self.indices] * self.bc_values[self.indices] + (1-self.bc_mask[self.indices])*self.u[self.indices]
		
		if self.tell_loss:
			l = loss(u,self.f[self.indices],
				self.bc_mask[self.indices],self.bc_values[self.indices])
		
		
		# update hidden states
		for i,index in enumerate(self.indices):
			self.hidden_states[index] = hidden_states[i]
			if self.tell_loss and l[i] > 20000:
				self.reset_env(index)
					
		# reset environments eventually
		self.step += 1
		if self.step % (self.average_sequence_length/self.batch_size) == 0:
			self.reset_env(int(self.reset_i))
			self.reset_i = (self.reset_i+1)%self.dataset_size
		
		if self.tell_loss:
			return torch.mean(l)
