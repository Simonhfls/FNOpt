import torch
from utils import has_nan

# this script contains multistep dataset functionality to:
# transform datasets with multiple channels into dataset with one channel (this is needed to concatenate multiple datasets with different numbers of channels)
# concatenate multiple datasets

class DatasetToSingleChannel:
	# transform datasets with multiple channels into dataset with one channel 
	# (this is needed to concatenate multiple datasets with different numbers of channels)
	
	def __init__(self,multi_channel_dataset):
		"""
		:multi_channel_dataset: dataset with multiple channels (e.g. cloth with x/y/z or fluid with a/p channels)
		"""
		self.multi_channel_dataset = multi_channel_dataset
	
	def ask(self):
		"""
		ask for a batch from multi_channel_dataset. The multi channel samples are split into single channel samples.
		:return: 
			:grads: gradients for accelerations (shape: batch_size*n_channels x 1 x h x w)
			:hidden_states: list of length batch_size * n_channels that contains the hidden_states 
							(list entries are None if corresponding hidden_states are not yet set)
		"""
		grads, hidden_states = self.multi_channel_dataset.ask()
		self.bs, self.c, self.h, self.w = grads.shape
		split_grads = grads.reshape(self.bs*self.c,1,self.h,self.w)
		hidden_states = [[None for _ in range(self.c)] if hs is None else hs for hs in hidden_states]
		split_hidden_states = [hs_split for hs_merged in hidden_states for hs_split in hs_merged]
		return split_grads, split_hidden_states

	def ask_sft(self):
		"""
		ask for a batch from multi_channel_dataset. The multi channel samples are split into single channel samples.
		:return:
			:grads: gradients for accelerations (shape: batch_size*n_channels x 1 x h x w)
			:hidden_states: list of length batch_size * n_channels that contains the hidden_states
							(list entries are None if corresponding hidden_states are not yet set)
		"""
		grads, hidden_states = self.multi_channel_dataset.ask_sft()
		self.bs, self.c, self.h, self.w = grads.shape
		split_grads = grads.reshape(self.bs*self.c,1,self.h,self.w)
		hidden_states = [[None for _ in range(self.c)] if hs is None else hs for hs in hidden_states]
		split_hidden_states = [hs_split for hs_merged in hidden_states for hs_split in hs_merged]
		return split_grads, split_hidden_states


	def tell(self,step, hidden_states=None):
		"""
		The single channel update steps and hidden_states are merged into multi channel updates.
		:step: update step for gradients given by ask(). (shape: batch_size*n_channels x 1 x h x w)
		:hidden_states: list of length batch_size * n_channels that contains the hidden_states that should be stored.
						This is useful to store e.g. momentum / variance / last update steps etc. 
						If None: no hidden_states are stored.
		:return: loss to optimize neural update-step-model
		"""
		merge_step = step.reshape(self.bs,self.c,self.h,self.w)
		merge_hidden_states = None if hidden_states is None else [hidden_states[i*self.c:(i+1)*self.c] for i in range(self.bs)]
		l = self.multi_channel_dataset.tell(merge_step,merge_hidden_states)
		return l

	def tell_sft(self, step, hidden_states=None):
		merge_step = step.reshape(self.bs, self.c, self.h, self.w)
		merge_hidden_states = None if hidden_states is None else [hidden_states[i * self.c:(i + 1) * self.c] for i in
																  range(self.bs)]
		l = self.multi_channel_dataset.tell_sft(merge_step, merge_hidden_states)
		return l



class DatasetConcat: # TODO
	# transform datasets with multiple channels into dataset with one channel 
	# (this is needed to concatenate multiple datasets with different numbers of channels)
	
	def __init__(self,datasets,logger=None,names=None,steps_per_log=1):
		"""
		:logger: logger to log the logarithm of the mean absolute gradients (LMAG) of the different datasets
		:names: names to log the different datasets
		"""
		self.datasets = datasets
		self.logger = logger
		self.names = names if (names is not None and logger is not None) else [f"{i}" for i in range(len(datasets))]
		self.steps=0
		self.steps_per_log = steps_per_log
		
	
	def ask(self):
		"""
		:return: 
			:grads: gradients for accelerations (shape: batch_size x 3 x h x w)
			:hidden_states: for optimizer
		"""
		results = [ds.ask() for ds in self.datasets]
		
		grads = torch.cat([r[0] for r in results],0)
		
		# log if logger is not None
		if self.logger is not None and self.steps%self.steps_per_log==0:
			for r,name,ds in zip(results,self.names,self.datasets):
				if type(ds)==DatasetToSingleChannel:
					ds = ds.multi_channel_dataset
				self.logger.log(f"LAG {name}",torch.mean(torch.log(torch.mean(torch.abs(r[0]),[1,2,3]))),self.steps) # mean log mean abs gradients
				self.logger.log(f"Itr {name}",torch.mean(ds.iterations[ds.indices]),self.steps) # mean #iterations
			self.logger.log("LAG total",torch.mean(torch.log(torch.mean(torch.abs(grads),[1,2,3]))),self.steps)
		self.steps += 1
		
		
		# extract batch sizes so everything can be properly put together afterwards again....
		self.batch_sizes = [len(r[1]) for r in results]
		hidden_states = [x for r in results for x in r[1]]
		
		return grads, hidden_states
	
	def tell(self,step, hidden_states=None):
		"""
		:step: update step for gradients given by ask()
		:return: loss to optimize neural update-step-model
		"""
		loss = 0
		i = 0
		for bs,dataset in zip(self.batch_sizes,self.datasets):
			loss_i = dataset.tell(step[i:(i+bs)],hidden_states[i:(i+bs)])
			loss = loss + torch.mean(loss_i)
			i += bs
			if has_nan(loss_i):
				print(f"loss_i {i} has nan!")
		
		return loss
