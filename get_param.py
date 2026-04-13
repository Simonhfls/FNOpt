import argparse
import os
from pathlib import Path

import torch
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig

def update_params(params):
	params.cloth.min_stretching = params.cloth.max_stretching if params.cloth.min_stretching is None else params.cloth.min_stretching
	params.cloth.min_shearing = params.cloth.max_shearing if params.cloth.min_shearing is None else params.cloth.min_shearing
	params.cloth.min_bending = params.cloth.max_bending if params.cloth.min_bending is None else params.cloth.min_bending
	params.cloth.min_a_ext = params.cloth.a_ext if params.cloth.min_a_ext is None else params.cloth.min_a_ext
	params.cloth.stretching_range = [params.cloth.min_stretching, params.cloth.max_stretching]
	params.cloth.shearing_range = [params.cloth.min_shearing, params.cloth.max_shearing]
	params.cloth.bending_range = [params.cloth.min_bending, params.cloth.max_bending]
	params.cloth.a_ext_range = [params.cloth.min_a_ext, params.cloth.a_ext]

	params.cloth.l_stretching = params.cloth.max_stretching if params.cloth.l_stretching is None else params.cloth.l_stretching
	params.cloth.l_shearing = params.cloth.max_shearing if params.cloth.l_shearing is None else params.cloth.l_shearing
	params.cloth.l_bending = params.cloth.max_bending if params.cloth.l_bending is None else params.cloth.l_bending
	# params.cloth.l_g = params.cloth.g if params.cloth.l_g is None else params.cloth.l_g
	params.cloth.l_L_0 = params.cloth.L_0 if params.cloth.l_L_0 is None else params.cloth.l_L_0
	params.cloth.l_dt = params.cloth.dt if params.cloth.l_dt is None else params.cloth.l_dt
	params.ema.update_after_step = params.data.n_batches_per_epoch * 10 if params.ema.update_after_step is None else params.ema.update_after_step

	if params.net.dtype == "float16":
		params.net.dtype = torch.float16
	elif params.net.dtype == "float32":
		params.net.dtype = torch.float32
	elif params.net.dtype == "float64":
		params.net.dtype = torch.float64
	torch.set_default_dtype(params.net.dtype)

	return params

def update_params_inference(params):
	params.cloth.stretching_range = [params.cloth.min_stretching, params.cloth.max_stretching]
	params.cloth.shearing_range = [params.cloth.min_shearing, params.cloth.max_shearing]
	params.cloth.bending_range = [params.cloth.min_bending, params.cloth.max_bending]
	params.cloth.a_ext_range = [params.cloth.min_a_ext, params.cloth.a_ext]

	if params.net.dtype == "float16":
		params.net.dtype = torch.float16
	elif params.net.dtype == "float32":
		params.net.dtype = torch.float32
	elif params.net.dtype == "float64":
		params.net.dtype = torch.float64
	torch.set_default_dtype(params.net.dtype)
	return params

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


def get_hyperparam(params):
	if params.net.name == "SymmetricMetamizer":
		return f"net {params.net}; sg: {params.symmetry_group}; hs {params.hidden_size}; dt {params.dt};"
	if params.net.name in ("FNOVertex", "UNOVertex"):
		return f"net {params.net.name}; hs {params.net.hidden_channels}; dt {params.cloth.dt};"
	if params.net.name == "MeshGraphNets":
		return f"net {params.net.name}; hs {params.flags.message_passing_aggregator}; mp {params.flags.message_passing_steps};"
	if params.net.name == "MeshGraphNets2":
		return f"net {params.net.name}; mp {params.net.message_passing_steps};"
	return f"net {params.net.name}; hs {params.net.hidden_size}; dt {params.cloth.dt};"

def get_load_hyperparam(params):
	if params.net.name == "SymmetricMetamizer":
		return f"net {params.net}; sg: {params.symmetry_group}; hs {params.hidden_size}; dt {params.dt};"
	if params.net.name in ("FNOVertex", "UNOVertex"):
		return f"net {params.net.name}; hs {params.net.hidden_channels}; dt {params.cloth.dt};"
	if params.net.name == "MeshGraphNets":
		return f"net {params.net.name}; hs {params.flags.message_passing_aggregator}; mp {params.flags.message_passing_steps};"
	if params.net.name == "MeshGraphNets2":
		return f"net {params.net.name}; mp {params.net.message_passing_steps};"
	return f"net {params.net.name}; hs {params.net.hidden_size}; dt {params.cloth.dt};"

cuda = torch.cuda.is_available()
print('cuda available:', cuda)

def toCuda(x):
	if type(x) is tuple or type(x) is list:
		return [xi.cuda() if cuda else xi for xi in x]
	return x.cuda() if cuda else x

def toCpu(x):
	if type(x) is tuple or type(x) is list:
		return [xi.detach().cpu() for xi in x]
	return x.detach().cpu()

def toDType(x):
	if type(x) is tuple or type(x) is list:
		return [xi.type(params.net.dtype) for xi in x]
	return x.type(params.net.dtype)


def get_params():
	config_folder = os.path.join(Path(__file__).parent, './configs')
	pipe = ConfigPipeline(
		[
			YamlConfig(
				# 'fno_vertex.yaml', config_name='local', config_folder=config_folder
				# 'fno_vertex.yaml', config_name='local_sft', config_folder=config_folder
				# 'uno_vertex.yaml', config_name='local', config_folder=config_folder
				# 'metamizer.yaml', config_name='local', config_folder=config_folder
				# 'mgnrp.yaml', config_name='local', config_folder=config_folder
				# 'meshgraphnets2.yaml', config_name='local', config_folder=config_folder
				# 'test_solver.yaml', config_name='local', config_folder=config_folder
				'abl_optimizers.yaml', config_name='local', config_folder=config_folder

	),
			ArgparseConfig(infer_types=True, config_name=None, config_file=None),
			YamlConfig(config_folder=config_folder),
			ArgparseConfig(infer_types=True, config_name=None, config_file=None),
		]
	)
	params = pipe.read_conf()
	# params = update_params(params)
	return params


params = get_params()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False