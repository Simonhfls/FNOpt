import os
from dataset_cloth import DatasetCloth
from dataset_utils import DatasetToSingleChannel, DatasetConcat
from fnopt import get_Net
from Logger import Logger
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from get_param import params, toCuda, get_hyperparam, get_load_hyperparam, toDType, update_params
from neuralop.utils import count_model_params
from utils import has_nan
import wandb
from tqdm import tqdm


def wandb_init(opt, tags=None):
	if opt.wandb.log:
		print("Wandb init ...")
		wandb.init(
			dir=opt.wandb.root_dir,
			project='FNOpt',
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
	params = update_params(params)

	print('params:', params)

	fnopt = toCuda(get_Net(params))
	fnopt = toDType(fnopt)
	fnopt.train()
	#fnopt.nn = torch.compile(fnopt.nn)

	n_params = count_model_params(fnopt)
	print(f'\nTotal number of parameters: {n_params}')

	steps_per_log = 10
	if params.opt.optimizer == 'adam':
		optimizer = Adam(fnopt.parameters(),lr=params.opt.lr)
	elif params.opt.optimizer == 'adamw':
		optimizer = AdamW(fnopt.parameters(),lr=params.opt.lr)
	scheduler = MultiStepLR(optimizer, milestones=[25,50,75], gamma=0.5)

	logger = Logger(get_hyperparam(params),use_csv=False,use_tensorboard=False)
	params.datetime = logger.datetime
	wandb_init(params)

	if params.cloth.load_latest or params.cloth.load_date_time is not None or params.cloth.load_index is not None:
		load_logger = Logger(get_load_hyperparam(params),use_csv=False,use_tensorboard=False)
		if params.cloth.load_optimizer:
			params.load_date_time, params.cloth.load_index = load_logger.load_state(fnopt,optimizer,params.cloth.load_date_time,params.cloth.load_index)
		else:
			params.load_date_time, params.cloth.load_index = load_logger.load_state(fnopt,None,params.cloth.load_date_time,params.cloth.load_index)
		params.cloth.load_index=int(params.cloth.load_index)
		print(f"loaded: {params.cloth.load_date_time}, {params.cloth.load_index}")
	params.cloth.load_index = 0 if params.cloth.load_index is None else params.cloth.load_index
	# fnopt = replace_with_periodic_padding(fnopt)	# TODO why no implementation for this?

	datasets = []
	names = []

	print('training dataset: cloth')

	for iterations_per_timestep in [1,3,10,30]:
		original_dataset_cloth = DatasetCloth(params.data.height,params.data.width,params.opt.batch_size,params.data.dataset_size,params.data.average_sequence_length,iterations_per_timestep=iterations_per_timestep,stiffness_range=params.cloth.stretching_range,shearing_range=params.cloth.shearing_range,bending_range=params.cloth.bending_range,a_ext_range=params.cloth.g)
		dataset_cloth = DatasetToSingleChannel(original_dataset_cloth)
		datasets.append(dataset_cloth)
		names.append(f"cloth_{iterations_per_timestep}")



	dataset = DatasetConcat(datasets,logger,names,steps_per_log)


	kwargs = {}

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


			update_steps, new_hidden_states = fnopt(grads, hidden_states, **kwargs)
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
				torch.nn.utils.clip_grad_value_(fnopt.parameters(), params.opt.clip_grad_value)
			if params.opt.clip_grad_norm is not None:
				torch.nn.utils.clip_grad_norm_(fnopt.parameters(), params.opt.clip_grad_norm)

			optimizer.step()

		if (epoch + 1) % params.opt.i_save == 0:
			# logger.save_state(cloth_net,optimizer,epoch+1)
			model_dir = 'Logger/{}/{}/states'.format(logger.name, logger.datetime)
			os.makedirs(model_dir, exist_ok=True)
			fnopt.model.save_checkpoint(savedir=model_dir, index=epoch + 1)
			print('model saved to:', model_dir)
			fnopt = toCuda(fnopt)

		scheduler.step()


