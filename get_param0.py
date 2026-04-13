import argparse
import torch

def str2bool(v):
	"""
	'boolean type variable' for add_argument
	"""
	if v.lower() in ('yes','true','t','y','1'):
		return True
	elif v.lower() in ('no','false','f','n','0'):
		return False
	else:
		raise argparse.ArgumentTypeError('boolean value expected.')

def get_params():
	"""
	return parameters for training / testing / plotting of models
	:return: parameter-Namespace
	"""
	parser = argparse.ArgumentParser(description='train / test a pytorch model to simulate cloth')
	parser.add_argument('--name', type=str, default='Metamizer_000', help='training run name')
	parser.add_argument('--wandb', default=False, type=str2bool, help='log models / metrics during training (wandb)')
	parser.add_argument('--wandb_root_dir',  type=str, default='./', help='wandb root directory')

	# Network parameters
	parser.add_argument('--net', default="Metamizer", type=str, help='network to train (default: Metamizer)', choices=["Metamizer","SymmetricMetamizer"])
	parser.add_argument('--SMP_model_type', default="Unet", type=str, help='model type used for SMP segmentation nets')
	parser.add_argument('--SMP_encoder_name', default="resnet34", type=str, help='encoder name used for SMP segmentation nets')
	parser.add_argument('--hidden_size', default=20, type=int, help='hidden size of network (default: 20)')
	parser.add_argument('--dtype', default="float64", type=str, help='data type (float32 or float 64) (default: float64)', choices=["float32","float64"])
	parser.add_argument('--symmetry_group', default='D4', type=str, help='symmetry group of equivariant CNN (default: D4)',choices=['C1','C2','C4','D1','D2','D4'])
	
	# Training parameters
	parser.add_argument('--n_epochs', default=100, type=int, help='number of epochs (after each epoch, the model gets saved)')
	parser.add_argument('--n_batches_per_epoch', default=500, type=int, help='number of batches per epoch (default: 5000)')
	# parser.add_argument('--iterations_per_timestep', default=1000, type=int, help='number of batches per epoch (default: 1000)') # should be much less later (maybe 3) or adaptive
	parser.add_argument('--iterations_per_timestep', default=10, type=int, help='number of batches per epoch (default: 1000)') # should be much less later (maybe 3) or adaptive
	parser.add_argument('--batch_size', default=3, type=int, help='batch size (default: 100)')
	parser.add_argument('--average_sequence_length', default=1000, type=int, help='average sequence length in dataset (default: 1000)')
	parser.add_argument('--dataset_size', default=500, type=int, help='size of dataset (default: 1000)')
	parser.add_argument('--cuda', default=False, type=str2bool, help='use GPU')
	parser.add_argument('--ema_beta', default=0.995, type=float, help='ema beta (default: 0.995)')
	parser.add_argument('--ema_update_after_step', default=None, type=int, help='only after this number of .update() calls will it start updating EMA (default: 100)')
	parser.add_argument('--ema_update_every', default=1, type=int, help='how often to actually update EMA, to save on compute (default: 1)')
	parser.add_argument('--lr', default=0.001, type=float, help='learning rate of optimizer (default: 0.001)')
	parser.add_argument('--clip_grad_norm', default=None, type=float, help='gradient norm clipping (default: None)')
	parser.add_argument('--clip_grad_value', default=None, type=float, help='gradient value clipping (default: None)')
	parser.add_argument('--plot', default=False, type=str2bool, help='plot during training')
	parser.add_argument('--log', default=True, type=str2bool, help='log models / metrics during training (turn off for debugging)')

	# Setup parameters
	parser.add_argument('--height', default=100, type=int, help='cloth height')
	parser.add_argument('--width', default=100, type=int, help='cloth width')
	
	# Cloth parameters
	parser.add_argument('--stiffness', default=1000, type=float, help='stiffness parameter of cloth')
	parser.add_argument('--min_stiffness', default=100, type=float, help='min stiffness range parameter of cloth (default: same as stiffness)')
	# parser.add_argument('--shearing', default=100, type=float, help='shearing parameter of cloth')
	parser.add_argument('--shearing', default=10, type=float, help='shearing parameter of cloth')
	parser.add_argument('--min_shearing', default=1, type=float, help='min shearing range parameter of cloth (default: same as shearing)')
	parser.add_argument('--bending', default=10, type=float, help='bending parameter of cloth')
	# parser.add_argument('--bending', default=0.01, type=float, help='bending parameter of cloth')
	parser.add_argument('--min_bending', default=0.01, type=float, help='min bending range parameter of cloth (default: same as bending)')
	parser.add_argument('--a_ext', default=1, type=float, help='gravitational constant (external acceleration)')
	parser.add_argument('--min_a_ext', default=None, type=float, help='min gravitational constant (default: same as a_ext)')
	parser.add_argument('--a_ext_noise_range', default=0, type=float, help='additional gauss noise for a_ext (default: 0)')
	parser.add_argument('--g', default=1, type=float, help='gravitational constant (deprecated: use a_ext instead!)')
	parser.add_argument('--L_0', default=1, type=float, help='rest length of cloth grid edges')
	parser.add_argument('--dt', default=1, type=float, help='timestep of cloth simulation integrator')
	
	# Fluid parameters
	parser.add_argument('--integrator', default='imex', type=str, help='integration scheme (explicit / implicit / imex) (default: imex)',choices=['explicit','implicit','imex'])
	parser.add_argument('--loss_bound', default=20, type=float, help='loss factor for boundary conditions')
	parser.add_argument('--loss_nav', default=1, type=float, help='loss factor for navier stokes equations')
	parser.add_argument('--loss_mean_a', default=0.1, type=float, help='loss factor to keep mean of a around 0')
	parser.add_argument('--loss_mean_p', default=0.1, type=float, help='loss factor to keep mean of p around 0')
	parser.add_argument('--regularize_grad_p', default=0, type=float, help='regularizer for gradient of p. evt needed for very high reynolds numbers (default: 0)')
	parser.add_argument('--max_speed', default=1, type=float, help='max speed for boundary conditions in dataset (default: 1)')
	parser.add_argument('--mu', default=10, type=float, help='mu parameter of fluid')
	parser.add_argument('--min_mu', default=0.01, type=float, help='min mu range parameter of fluid (if None: same as mu)')
	parser.add_argument('--rho', default=10, type=float, help='rho parameter of fluid')
	parser.add_argument('--min_rho', default=0.1, type=float, help='min rho range parameter of fluid (if None: same as rho)')
	
	# Diffusion Advection parameters
	parser.add_argument('--D', default=1000, type=float, help='Diffusivity parameter')
	parser.add_argument('--min_D', default=1, type=float, help='min Diffusivity range parameter (if None: same as D)')
	
	# wave equation parameters
	parser.add_argument('--c', default=10, type=float, help='wave propagation speed c parameter')
	parser.add_argument('--min_c', default=0.1, type=float, help='min wave propagation speed c range parameter (if None: same as D)')
	
	# Load parameters
	parser.add_argument('--l_stiffness', default=None, type=float, help='load stiffness parameter of cloth')
	parser.add_argument('--l_shearing', default=None, type=float, help='load shearing parameter of cloth')
	parser.add_argument('--l_bending', default=None, type=float, help='load bending parameter of cloth')
	parser.add_argument('--l_g', default=None, type=float, help='load gravitational constant')
	parser.add_argument('--l_L_0', default=None, type=float, help='load rest length of cloth grid edges')
	parser.add_argument('--l_dt', default=1, type=float, help='load timestep of cloth simulation integrator')
	parser.add_argument('--load_date_time', default=None, type=str, help='date_time of run to load (default: None)')
	parser.add_argument('--load_index', default=None, type=str, help='index of run to load (default: None)')
	#parser.add_argument('--load_index', default=None, type=int, help='index of run to load (default: None)')
	parser.add_argument('--load_optimizer', default=False, type=str2bool, help='load state of optimizer (default: True)')
	parser.add_argument('--load_latest', default=False, type=str2bool, help='load latest version for training (if True: leave load_date_time and load_index None. default: False)')


	# parse parameters
	params = parser.parse_args()
	
	params.min_stiffness = params.stiffness if params.min_stiffness is None else params.min_stiffness
	params.min_shearing = params.shearing if params.min_shearing is None else params.min_shearing
	params.min_bending = params.bending if params.min_bending is None else params.min_bending
	params.min_a_ext = params.a_ext if params.min_a_ext is None else params.min_a_ext
	params.stiffness_range = [params.min_stiffness,params.stiffness]
	params.shearing_range = [params.min_shearing,params.shearing]
	params.bending_range = [params.min_bending,params.bending]
	params.a_ext_range = [params.min_a_ext,params.a_ext]
	params.D_range = [params.min_D,params.D]
	
	params.l_stiffness = params.stiffness if params.l_stiffness is None else params.l_stiffness
	params.l_shearing = params.shearing if params.l_shearing is None else params.l_shearing
	params.l_bending = params.bending if params.l_bending is None else params.l_bending
	params.l_g = params.g if params.l_g is None else params.l_g
	params.l_L_0 = params.L_0 if params.l_L_0 is None else params.l_L_0
	params.l_dt = params.dt if params.l_dt is None else params.l_dt
	
	params.ema_update_after_step = params.n_batches_per_epoch*10 if params.ema_update_after_step is None else params.ema_update_after_step
	
	if params.dtype == "float16":
		params.dtype = torch.float16
	elif params.dtype == "float32":
		params.dtype = torch.float32
	elif params.dtype == "float64":
		params.dtype = torch.float64
	torch.set_default_dtype(params.dtype)
	
	return params

def get_hyperparam(params):
	if params.net == "SymmetricMetamizer":
		return f"net {params.net}; sg: {params.symmetry_group}; hs {params.hidden_size}; dt {params.dt};"
	return f"net {params.net}; hs {params.hidden_size}; dt {params.dt};"

def get_load_hyperparam(params):
	if params.net == "SymmetricMetamizer":
		return f"net {params.net}; sg: {params.symmetry_group}; hs {params.hidden_size}; dt {params.dt};"
	return f"net {params.net}; hs {params.hidden_size}; dt {params.dt};"

def toCuda(x):
	if type(x) is tuple or type(x) is list:
		return [xi.cuda() if params.cuda else xi for xi in x]
	return x.cuda() if params.cuda else x

def toCpu(x):
	if type(x) is tuple or type(x) is list:
		return [xi.detach().cpu() for xi in x]
	return x.detach().cpu()

def toDType(x):
	if type(x) is tuple or type(x) is list:
		return [xi.type(params.dtype) for xi in x]
	return x.type(params.dtype)

params = get_params()
device = 'cuda' if params.cuda else 'cpu'
