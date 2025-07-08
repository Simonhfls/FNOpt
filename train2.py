import os
import sys
sys.path.append(os.path.join(os.getcwd(), "../baseline/meshgraphnets"))
from dataset_cloth3 import DatasetCloth
from dataset_poisson import DatasetPoisson
from dataset_fluid import DatasetFluid
from dataset_diffusion import DatasetDiffusion
from dataset_utils import DatasetToSingleChannel, DatasetConcat
from metamizer import get_Net3 as get_Net
from Logger import Logger
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from get_param2 import params, toCuda, get_hyperparam, get_load_hyperparam, toDType
from neuralop.utils import count_model_params
from utils import has_nan
import wandb
from tqdm import tqdm


def wandb_init(opt, tags=None):
	if opt.wandb.log:
		print("Wandb init ...")
		wandb.init(
			dir=opt.wandb.root_dir,
			project='Metamizer',
			name=opt.name,
			mode='offline',
			config=opt,
			tags=tags
		)
		print('wandb dir:', wandb.run.dir)
	print('run name:', opt.name)


if __name__ == '__main__':
	torch.manual_seed(0)
	torch.set_num_threads(4)
	np.random.seed(0)
	torch.set_float32_matmul_precision('high')
	params.training = True
	print('params:', params)

	metamizer = toCuda(get_Net(params))
	metamizer = toDType(metamizer)
	metamizer.train()
	#metamizer.nn = torch.compile(metamizer.nn)

	n_params = count_model_params(metamizer)
	print(f'\nTotal number of parameters: {n_params}')

	steps_per_log = 10
	if params.opt.optimizer == 'adam':
		optimizer = Adam(metamizer.parameters(),lr=params.opt.lr)
	elif params.opt.optimizer == 'adamw':
		optimizer = AdamW(metamizer.parameters(),lr=params.opt.lr)
	scheduler = MultiStepLR(optimizer, milestones=[25,50,75], gamma=0.5)
	# wandb_init(params)

	logger = Logger(get_hyperparam(params),use_csv=False,use_tensorboard=False)
	params.datetime = logger.datetime
	wandb_init(params)


	if params.cloth.load_latest or params.cloth.load_date_time is not None or params.cloth.load_index is not None:
		load_logger = Logger(get_load_hyperparam(params),use_csv=False,use_tensorboard=False)
		if params.cloth.load_optimizer:
			params.load_date_time, params.cloth.load_index = load_logger.load_state(metamizer,optimizer,params.cloth.load_date_time,params.cloth.load_index)
		else:
			params.load_date_time, params.cloth.load_index = load_logger.load_state(metamizer,None,params.cloth.load_date_time,params.cloth.load_index)
		params.cloth.load_index=int(params.cloth.load_index)
		print(f"loaded: {params.cloth.load_date_time}, {params.cloth.load_index}")
	params.cloth.load_index = 0 if params.cloth.load_index is None else params.cloth.load_index

	# metamizer = replace_with_periodic_padding(metamizer)	# TODO why no implementation for this?

	datasets = []
	names = []

	print('training dataset:', end=' ')
	for ds in ['cloth', 'fluid', 'diffusion', 'poisson']:
		if params.data[f"use_{ds}"] == True:
			print(ds, end=' ')
	print()

	for iterations_per_timestep in [1,3,10,30]:
	#for iterations_per_timestep in [5,25]:

		if params.data.use_cloth:
			# cloth dataset
			# if params.cloth.use_f_ext:
			# 	original_dataset_cloth = DatasetClothWithExternalForce(params.data.height,params.data.width,params.opt.batch_size,params.data.dataset_size,params.data.average_sequence_length,iterations_per_timestep=iterations_per_timestep,stiffness_range=params.cloth.stretching_range,shearing_range=params.cloth.shearing_range,bending_range=params.cloth.bending_range,a_ext_range=params.cloth.g)
			# else:
			original_dataset_cloth = DatasetCloth(params.data.height,params.data.width,params.opt.batch_size,params.data.dataset_size,params.data.average_sequence_length,iterations_per_timestep=iterations_per_timestep,stiffness_range=params.cloth.stretching_range,shearing_range=params.cloth.shearing_range,bending_range=params.cloth.bending_range,a_ext_range=params.cloth.g)

			#dataset = original_dataset
			dataset_cloth = DatasetToSingleChannel(original_dataset_cloth)
			datasets.append(dataset_cloth)
			names.append(f"cloth_{iterations_per_timestep}")

		if params.data.use_fluid:
			# fluid dataset
			original_dataset_fluid = DatasetFluid(params.data.height,params.data.width,params.opt.batch_size,params.data.dataset_size,params.data.average_sequence_length,iterations_per_timestep=iterations_per_timestep)
			dataset_fluid = DatasetToSingleChannel(original_dataset_fluid)
			datasets.append(dataset_fluid)
			names.append(f"fluid_{iterations_per_timestep}")

		if params.data.use_diffusion:
			# diffusion dataset
			original_dataset_diffusion = DatasetDiffusion(params.data.height,params.data.width,params.opt.batch_size,params.data.dataset_size,average_sequence_length=200,iterations_per_timestep=iterations_per_timestep)
			dataset_diffusion = DatasetToSingleChannel(original_dataset_diffusion)
			datasets.append(dataset_diffusion)
			names.append(f"diffusion_{iterations_per_timestep}")

	if params.data.use_poisson:
		# poisson dataset
		#dataset_poisson = DatasetPoisson(params.data.height,params.data.width,params.opt.batch_size*2,params.data.dataset_size,average_sequence_length=60)
		dataset_poisson = DatasetPoisson(params.data.height,params.data.width,params.opt.batch_size,params.data.dataset_size,average_sequence_length=60)
		datasets.append(dataset_poisson)
		names.append(f"laplace")

	dataset = DatasetConcat(datasets,logger,names,steps_per_log)
	#dataset = DatasetConcat([dataset_cloth,dataset_poisson,dataset_fluid])
	#dataset = DatasetConcat([dataset_cloth,dataset_poisson])
	#dataset = DatasetConcat([dataset_fluid])


	for epoch in range(int(params.cloth.load_index), params.opt.n_epochs):
		print(f"Epoch: {epoch} / {params.opt.n_epochs}")

		pbar = tqdm(range(params.data.n_batches_per_epoch), desc=f"Epoch {epoch}", unit="batch")
		for step in pbar:
			grads, hidden_states = dataset.ask()
			if has_nan(grads):
				print("input grads contain nan!")
				exit()
			for hs in hidden_states:
				if hs is not None:
					for h in hs:
						if has_nan(h):
							print("input hidden_states contain nan!")
							exit()

			update_steps, new_hidden_states = metamizer(grads, hidden_states)
			if has_nan(update_steps):
				print("output update_steps contain nan!")
				exit()
			for hs in new_hidden_states:
				for h in hs:
					if has_nan(h):
						print("output hidden_states contain nan!")
						exit()

			loss = dataset.tell(update_steps, new_hidden_states)

			if step % steps_per_log == 0:
				logger.log("L", loss, epoch * params.data.n_batches_per_epoch + step)

			pbar.set_postfix(loss=loss.item())

			if has_nan(loss):
				print("loss has nan!")

			optimizer.zero_grad()
			loss.backward()

			if params.opt.clip_grad_value is not None:
				torch.nn.utils.clip_grad_value_(metamizer.parameters(), params.opt.clip_grad_value)
			if params.opt.clip_grad_norm is not None:
				torch.nn.utils.clip_grad_norm_(metamizer.parameters(), params.opt.clip_grad_norm)

			optimizer.step()

		if (epoch + 1) % params.opt.i_save == 0:
			if not params.net.name == 'MeshGraphNets2':
				logger.save_state(metamizer.cpu(), optimizer, epoch + 1)
			else:
				# logger.save_state(cloth_net,optimizer,epoch+1)
				model_dir = 'Logger/{}/{}/states'.format(logger.name, logger.datetime)
				os.makedirs(model_dir, exist_ok=True)
				metamizer.model.save_checkpoint(savedir=model_dir, index=epoch + 1)
				print('model saved to:', model_dir)
			metamizer = toCuda(metamizer)

		scheduler.step()

# beispiel command:
	# python train_multistep.py --log=f --net=Grad_net_tiny --cuda=f --batch_size=1 --dataset_size=1 --n_batches_per_epoch=1000
	# python train_multistep.py --log=f --net=Grad_net_scale_inv --cuda=f --batch_size=10 --dataset_size=100 --n_batches_per_epoch=1000 --plot=t --iterations_per_timestep=10
	# python train_multistep.py --log=f --net=Grad_net_scale_inv --cuda=f --batch_size=10 --dataset_size=100 --n_batches_per_epoch=1000 --plot=t --iterations_per_timestep=10
	# python train_multistep.py --net=Grad_net_scale_inv --batch_size=10 --dataset_size=100 --n_batches_per_epoch=1000 --iterations_per_timestep=10 --cuda=f



