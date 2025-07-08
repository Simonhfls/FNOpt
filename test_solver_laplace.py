"""
sorry, this script is a bit messy...
This script compares different solvers at solving the poisson equation. The height and width of the setup is specified by --height and --width parameters.
To reproduce a specific comparison of Metamizer with other solvers, please uncomment the corresponding code sections below (see Lines 354-476)
"""
from dataset_poisson import DatasetPoisson
from get_param2 import params,toCuda,toCpu,get_hyperparam,toDType, device
import matplotlib.pyplot as plt
# from dataset_poisson import DatasetPoisson
from metamizer import get_Net3 as get_Net
from metamizer import Metamizer
from Logger import Logger
import torch
import numpy as np
from derivatives import laplace,laplace_detach
import scipy.sparse as sp
import scipy.sparse.linalg as la
import scipy
import time
from torch.optim import SGD,Adam,AdamW,RMSprop,Adagrad,Adadelta
from metamizer import Metamizer

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
print('height:', params.data.height, 'width:', params.data.width)
 
def loss(x,bc_mask):
	x = toDType(toCuda(torch.tensor(x).reshape(1,1,params.data.height,params.data.width)))
	x = bc_mask*dataset.bc_values + (1-bc_mask)*x
	
	residuals = laplace(x)
	loss = torch.sum(residuals**2*(1-bc_mask))/torch.sum(1-bc_mask)
	return loss.cpu().numpy()

def loss_gt(x,x_gt,bc_mask):
	x = toDType(toCuda(torch.tensor(x).reshape(1,1,params.data.height,params.data.width)))
	x = bc_mask*dataset.bc_values + (1-bc_mask)*x
	x_gt = x_gt.reshape(1,1,params.data.height,params.data.width)
	
	loss = torch.sum((x-x_gt)**2*(1-bc_mask))/torch.sum(1-bc_mask)
	return loss.cpu().numpy()
	

def laplace_to_matrix(input_bc, mask_bc):
	N = input_bc.shape[0]

	A = sp.diags(-12*np.ones(N*N),0)
	left = sp.diags(2*np.ones(N*N-1), -1)
	right = sp.diags(2*np.ones(N*N-1), 1)
	top = sp.diags(2*np.ones(N*(N-1)), -N)
	bottom = sp.diags(2*np.ones(N*(N-1)), N)
	topleft = sp.diags(1*np.ones(N*(N-1)-1), -N-1)
	topright = sp.diags(1*np.ones(N*(N-1)+1), -N+1)
	botleft = sp.diags(1*np.ones(N*(N-1)+1), N-1)
	botright = sp.diags(1*np.ones(N*(N-1)-1), N+1)
	
	A += left + right + top + bottom + topleft + topright + botleft + botright
	A *= 0.25

	b = input_bc.reshape(N*N)

	x,y = (mask_bc > 0).nonzero()

	A = sp.csr_matrix(A)
	
	for row in list(x*N+y):
		start = A.indptr[row]
		end = A.indptr[row + 1]
		# Set all entries in this row to zero
		A.data[start:end] = 0
	A.eliminate_zeros()
	A[x*N+y,x*N+y] = 1
	
	A = sp.csc_matrix(A)
	return A, b

def laplace_to_symmetric_matrix(input_bc, mask_bc):
	N = input_bc.shape[0]

	A = sp.diags(-12*np.ones(N*N),0)
	left = sp.diags(2*np.ones(N*N-1), -1)
	right = sp.diags(2*np.ones(N*N-1), 1)
	top = sp.diags(2*np.ones(N*(N-1)), -N)
	bottom = sp.diags(2*np.ones(N*(N-1)), N)
	topleft = sp.diags(1*np.ones(N*(N-1)-1), -N-1)
	topright = sp.diags(1*np.ones(N*(N-1)+1), -N+1)
	botleft = sp.diags(1*np.ones(N*(N-1)+1), N-1)
	botright = sp.diags(1*np.ones(N*(N-1)-1), N+1)
	
	A += left + right + top + bottom + topleft + topright + botleft + botright
	A *= -0.25#0.25

	b = input_bc.reshape(N*N)

	x,y = (mask_bc > 0).nonzero()

	A = sp.csr_matrix(A)
	
	# set rows to 0
	for row in list(x*N+y):
		start = A.indptr[row]
		end = A.indptr[row + 1]
		# Set all entries in this row to zero
		A.data[start:end] = 0
		
	A.eliminate_zeros()
			
	# set columns to 0
	A = A.tocsc()
	for col in list(x*N+y):
		start = A.indptr[col]
		end = A.indptr[col + 1]
		A.data[start:end] = 0

	# Step 4: Eliminate explicit zeros again
	A.eliminate_zeros()

	# Step 5: Convert back to CSR and set diagonal elements to 1
	A = A.tocsr()
		
	A[x*N+y,x*N+y] = 1
	
	A = sp.csc_matrix(A)
	return A, b

# neural solver (Metamizer)
def neural_solver(maxiter):
	#with torch.cuda.amp.autocast(): # deprecated and actually slowed down code
	with torch.no_grad():
		dataset.u[:] = 0
		dataset.hidden_states[0] = None
		for i in range(maxiter):
			grads, hidden_states = dataset.ask()
			update_steps, new_hidden_states = metamizer(grads, hidden_states)
			dataset.tell(update_steps, new_hidden_states)
	return dataset.u[0,0].cpu().numpy()


# gradient descent solver
def generate_solver(optimizer,*args):
	def grad_desc_solver(A,b,maxiter):
		dataset.u[:] = 0
		dataset.hidden_states[0] = None
		x = torch.zeros(1,1,params.data.height,params.data.width,requires_grad = True,device=device)
		optim = optimizer([x],*args)
		
		for i in range(maxiter):
			optim.zero_grad()
			grads, hidden_states = dataset.ask()
			x_old = x.data.clone()
			x.grad = grads
			optim.step()
			dataset.tell(x.data-x_old)
		
		return dataset.u[0,0].cpu().numpy()
	
	return grad_desc_solver

# scipy optimizer than directly aim to minimize the residuals (and get gradient information but not the matrix A)
def scipy_optimizer(maxiter,method):
	dataset.u[:] = 0
	dataset.hidden_states[0] = None
	def jacobian(x):
		# convert x to u
		dataset.u[0,:] = toCuda(torch.tensor(x)).reshape(1,params.data.height,params.data.width)
		grads, _ = dataset.ask()
		# convert grads to jacobian
		jac = toCpu(grads.reshape(params.data.height*params.data.width)).numpy()
		return jac
	
	def fun(x):
		return loss(x,dataset.bc_mask)
	
	x0 = np.zeros(params.data.height*params.data.width)
	x = scipy.optimize.minimize(fun=fun,x0=x0,method=method,jac=jacobian,tol=1e-40,options={"maxiter":maxiter,"maxcor":2})
	#print(f"x: {x}")
	return x.x

# PyAMG solvers
from pyamg import smoothed_aggregation_solver
import pyamg

def pyamg_optimizer(A,b,x0,maxiter,rtol):
	
	# Build a multigrid preconditioner using PyAMG
	ml = smoothed_aggregation_solver(A)
	M = ml.aspreconditioner()

	# Solve the system using Conjugate Gradient (CG) with the multigrid preconditioner
	#x, info = la.cg(A, b, x0=x0, rtol=rtol, maxiter=maxiter, M=M) # TODO: replace by pyamg cg
	x, info = pyamg.krylov.cg(A, b, x0=x0, tol=rtol, maxiter=maxiter, M=M) # TODO: replace by pyamg cg
	return x

# Scipy CG solvers
# from scipy.sparse.linalg import spilu, LinearOperator
# from sksparse.cholmod import cholesky
# def scipy_cg_solver(A,b,x0,maxiter,rtol,preconditioner):
#
# 	if preconditioner=="AMG":
# 		# Build a multigrid preconditioner using PyAMG
# 		ml = smoothed_aggregation_solver(A)
# 		M = ml.aspreconditioner()
# 	elif preconditioner=="ILU":
# 		# Build an ILU preconditioner
# 		ilu = spilu(A,fill_factor=30)  # Incomplete LU factorization
# 		M = LinearOperator(A.shape, matvec=lambda x: ilu.solve(x))  # Wrap as a linear operator
# 	elif preconditioner=="IC":
# 		# Build an IC preconditioner
# 		chol = cholesky(A)  # Incomplete Cholesky factorization
# 		M = LinearOperator(A.shape, matvec=lambda x: chol(x))
#
# 	x, info = la.cg(A, b, x0=x0, rtol=rtol, maxiter=maxiter, M=M) # TODO: replace by pyamg cg
# 	return x

"""
# PyAMGx solver (on GPU)
# to install pyamgx, follow: https://pyamgx.readthedocs.io/en/latest/install.html
# note, that this implementation wastes a lot of time during initialization and destruction of resources.
# the pure runtime of the solving algorithm is actually faster than metamizer.
# However, metamizer is more generally applicable e.g. also to nonlinear PDEs...
import pyamgx
import json

pyamgx_config_buffer = None
pyamgx_cfg, pyamgx_rsc, pyamgx_d = None, None, None

def pyamgx_optimizer(A,b,x0,maxiter,rtol,config="AGGREGATION_GS"):
	global pyamgx_config_buffer, pyamgx_cfg, pyamgx_rsc, pyamgx_d
	global A_amgx, b_amgx, x_amgx
	
	pyamgx.initialize()
	with open(f'/home/wandeln/Projects/AMGX/amgx/src/configs/{config}.json') as f:
		pyamgx_d = json.load(f)
	
	print(pyamgx_d)
	
	# TODO: modify maxiter / rtol in config dict...
	pyamgx_d["solver"]["max_iters"] = maxiter
	pyamgx_d["solver"]["tolerance"] = rtol
	#pyamgx_d["solver"]["print_grid_stats"] = 0
	#pyamgx_d["solver"]["print_solve_stats"] = 0
	#pyamgx_d["solver"]["obtain_timings"] = 0
	#pyamgx_d["solver"]["monitor_residual"] = 0
	

	pyamgx_cfg = pyamgx.Config().create_from_dict(pyamgx_d)

	pyamgx_rsc = pyamgx.Resources().create_simple(pyamgx_cfg) # => takes a lot of time
	
	# Create solver:
	solver_amgx = pyamgx.Solver().create(pyamgx_rsc, pyamgx_cfg)
	
	# Create matrices and vectors:
	A_amgx = pyamgx.Matrix().create(pyamgx_rsc)
	b_amgx = pyamgx.Vector().create(pyamgx_rsc)
	x_amgx = pyamgx.Vector().create(pyamgx_rsc)
	
	for i in range(1):
		
		# Upload system:
		A_amgx.upload_CSR(A)
		b_amgx.upload(b)
		x_amgx.upload(x0)
		
		# Setup and solve system:
		solver_amgx.setup(A_amgx)
		
		b_amgx.upload(b)
		x_amgx.upload(x0)
		solver_amgx.solve(b_amgx, x_amgx)
		
		# Download solution
		sol = np.zeros(params.data.height*params.data.width)
		x_amgx.download(sol)
	
	# Clean up:
	A_amgx.destroy()
	x_amgx.destroy()
	b_amgx.destroy()
	solver_amgx.destroy()
	
	pyamgx_rsc.destroy() # => takes a lot of time
	pyamgx_cfg.destroy()
	
	pyamgx.finalize()
	
	return sol
"""

# load neural net
metamizer = toDType(toCuda(get_Net(params)))
#metamizer.nn = torch.compile(metamizer.nn)
logger = Logger(get_hyperparam(params),use_csv=False,use_tensorboard=False)
date_time,index = logger.load_state(metamizer,None,datetime=params.inference.load_date_time,index=params.inference.load_index)
print(f"loaded: {date_time}, {index}")
metamizer.eval()
#metamizer.nn = torch.compile(metamizer.nn)


dataset = DatasetPoisson(params.data.height,params.data.width,1,1,average_sequence_length=9999999999999999,tell_loss=False)
dataset.reset0_env(0)
#dataset.reset1_env(0)

x_gt = (dataset.x_mesh/dataset.w)*(dataset.y_mesh/dataset.h)

bc_values, bc_mask = dataset.bc_values[0,0].cpu().numpy(), dataset.bc_mask[0,0].cpu().numpy()
bc_values = bc_values*bc_mask


print("generating sparse matrix...")
start = time.time()
A,b = laplace_to_matrix(bc_values, bc_mask)
x0 = np.zeros((params.data.height*params.data.width))
print(f"done after {time.time()-start} s")


# cupy version test
import cupy
import cupyx
import cupyx.scipy.sparse
import cupyx.scipy.sparse.linalg

# apply laplace to bc_values to achieve symmetric values
cuda_bc_values = toCuda(torch.tensor(bc_values).unsqueeze(0).unsqueeze(0))
laplace_cuda_bc_values = -laplace(cuda_bc_values).detach()
laplace_bc_values = -laplace_cuda_bc_values[0,0].cpu().numpy()
laplace_bc_values = bc_mask*bc_values+(1-bc_mask)*laplace_bc_values

A,b = laplace_to_symmetric_matrix(laplace_bc_values, bc_mask)

#np.set_printoptions(threshold = np.inf)
#print(A.diagonal())
#print(np.all(A.diagonal()==1)) # die diagonale von A ist doch nicht die identity matrix?
#exit()

x0 = np.zeros((params.data.height*params.data.width))
cp_A = cupyx.scipy.sparse.csc_matrix(A.copy())
cp_A_diag = cupyx.scipy.sparse.diags(cp_A.diagonal())
cp_A_diag_inv = cupyx.scipy.sparse.diags(cp_A.diagonal()**-1)
M = cp_A_diag_inv

cp_b = cupy.asarray(b.copy())
cp_x0 = cupy.asarray(x0.copy())

# custom number of iterations for different solvers

iterations = [1,1,2,5,10,20,50,100,200,500,1000,2000,5000,10000,20000,50000] # first step for "warm up" (sometimes the first step seems to take longer ... maybe to loading modules)

"""
title = "Metamizer vs gradient based solvers and scipy sparse solvers (deprecated)"
solvers = {"Metamizer": [lambda A,b,maxiter: neural_solver(maxiter), iterations[:10]]
		#,"SGD (lr=0.1)": [generate_solver(SGD,0.1), iterations[:13]] # diverges
		,"SGD (lr=0.01)": [generate_solver(SGD,0.01), iterations[:13]] # minimal loss: 2.41e-06
		#,"SGD (lr=0.001)": [generate_solver(SGD,0.001), iterations[:13]]
		#,"Adam (lr=0.1)": [generate_solver(Adam,0.1), iterations[:13]]
		,"Adam (lr=0.01)": [generate_solver(Adam,0.01), iterations[:13]] # minimal loss: 6.57e-07
		#,"Adam (lr=0.001)": [generate_solver(Adam,0.001), iterations[:13]]
		,"AdamW (lr=0.01)": [generate_solver(AdamW,0.01), iterations[:13]] # minimal loss: 7.18e-07
		,"RMSprop (lr=0.01)": [generate_solver(RMSprop,0.01), iterations[:13]] # minimal loss: 8.23e-05
		,"Adagrad (lr=0.1)": [generate_solver(Adagrad,0.1), iterations[:13]] # minimal loss: 2.77e-07
		,"Adadelta (lr=0.1)": [generate_solver(Adadelta,0.1), iterations[:13]] # minimal loss: 9.71e-03
		,"cg": [lambda A,b,maxiter: la.cg(A.copy(), b.copy(), x0.copy(), maxiter = maxiter, tol=1e-34)[0], iterations[:10]]
		#,"cgs": [lambda A,b,maxiter: la.cgs(A.copy(), b.copy(), x0.copy(), maxiter = maxiter, tol=1e-34)[0], iterations[:10]]
		#,"bicg": [lambda A,b,maxiter: la.bicg(A.copy(), b.copy(), x0.copy(), maxiter = maxiter, tol=1e-34)[0], iterations[:10]]
		#,"bicgstab": [lambda A,b,maxiter: la.bicgstab(A.copy(), b.copy(), x0.copy(), maxiter = maxiter, tol=1e-34)[0], iterations[:10]] # didn't converge
		,"minres": [lambda A,b,maxiter: la.minres(A.copy(), b.copy(), x0.copy(), maxiter = maxiter, tol=1e-34)[0], iterations[:14]]
		#,"qmr": [lambda A,b,maxiter: la.qmr(A.copy(), b.copy(), x0.copy(), maxiter = maxiter, tol=1e-34)[0]], iterations[:10]] # didn't converge
		#,"tfqmr": [lambda A,b,maxiter: la.tfqmr(A.copy(), b.copy(), x0.copy(), maxiter = maxiter, tol=1e-34)[0]], iterations[:10]] # didn't converge
		,"gcrotmk": [lambda A,b,maxiter: la.gcrotmk(A.copy(), b.copy(), x0.copy(), maxiter = maxiter, tol=1e-34)[0], iterations[:10]]
		,"gmres": [lambda A,b,maxiter: la.gmres(A.copy(), b.copy(), x0.copy(), maxiter = maxiter, tol=1e-34)[0], iterations[:10]]
		,"lgmres": [lambda A,b,maxiter: la.lgmres(A.copy(), b.copy(), x0.copy(), maxiter = maxiter, tol=1e-34)[0], iterations[:10]]}
"""

"""
# comparison of different learning rates
grad_optimizer = AdamW#Adagrad#Adam#SGD#Adadelta#
grad_optimizer_name = "AdamW"#"Adagrad"#"Adam"#"SGD"#"Adadelta"#
title = f"Metamizer and {grad_optimizer_name} (GPU) {params.data.height}x{params.data.width}"
solvers = {"Metamizer": [lambda A,b,maxiter: neural_solver(maxiter), iterations[:10]]}
solvers.update({f"{grad_optimizer_name} (lr={lr})": [generate_solver(grad_optimizer,lr), iterations[:13]] for lr in [0.1,0.03,0.01,0.003,0.001,0.0003,0.0001]}) # comparison of different learning rates
"""

"""
title = f"Metamizer and CuPy (GPU) Sparse Linear System Solvers {params.data.height}x{params.data.width}"
solvers = {"Metamizer": [lambda A,b,maxiter: neural_solver(maxiter), iterations[:10]]
		,"cuda_gmres": [lambda A,b,maxiter: cupyx.scipy.sparse.linalg.gmres(cp_A, cp_b, cp_x0, maxiter = maxiter, tol=1e-40)[0], iterations[:16]]
		,"cuda_lsmr": [lambda A,b,maxiter: cupyx.scipy.sparse.linalg.lsmr(cp_A, cp_b, cp_x0, maxiter = maxiter, atol=1e-40)[0], iterations[:16]]
		,"cuda_minres": [lambda A,b,maxiter: cupyx.scipy.sparse.linalg.minres(cp_A, cp_b, cp_x0, maxiter = maxiter, tol=1e-40)[0], iterations[:15]]
		,"cuda_cg": [lambda A,b,maxiter: cupyx.scipy.sparse.linalg.cg(cp_A, cp_b, cp_x0, maxiter = maxiter, tol=1e-40)[0], iterations[:13]]
		#,"cuda_cgs": [lambda A,b,maxiter: cupyx.scipy.sparse.linalg.cgs(cp_A, cp_b, cp_x0, maxiter = maxiter, tol=1e-40)[0], iterations[:13]]
		}
"""


title = f"Metamizer and GPU based Solvers {params.data.height}x{params.data.width}"
solvers = {params.net.name: [lambda A,b,maxiter: neural_solver(maxiter), iterations[:10]]
		# ,"SGD (lr=0.01)": [generate_solver(SGD,0.01), iterations[:13]] # minimal loss: 2.41e-06
		# ,"Adam (lr=0.01)": [generate_solver(Adam,0.01), iterations[:13]] # minimal loss: 6.57e-07
		# ,"AdamW (lr=0.01)": [generate_solver(AdamW,0.01), iterations[:13]] # minimal loss: 7.18e-07
		# ,"RMSprop (lr=0.01)": [generate_solver(RMSprop,0.01), iterations[:13]] # minimal loss: 8.23e-05
		# ,"Adagrad (lr=0.1)": [generate_solver(Adagrad,0.1), iterations[:13]] # minimal loss: 2.77e-07
		# ,"Adadelta (lr=0.1)": [generate_solver(Adadelta,0.1), iterations[:13]] # minimal loss: 9.71e-03
		# ,"cuda_gmres": [lambda A,b,maxiter: cupyx.scipy.sparse.linalg.gmres(cp_A, cp_b, cp_x0, maxiter = maxiter, tol=1e-40)[0], iterations[:16]]
		# ,"cuda_lsmr": [lambda A,b,maxiter: cupyx.scipy.sparse.linalg.lsmr(cp_A, cp_b, cp_x0, maxiter = maxiter, atol=1e-40)[0], iterations[:16]]
		# ,"cuda_minres": [lambda A,b,maxiter: cupyx.scipy.sparse.linalg.minres(cp_A, cp_b, cp_x0, maxiter = maxiter, tol=1e-40)[0], iterations[:15]]
		# ,"cuda_cg": [lambda A,b,maxiter: cupyx.scipy.sparse.linalg.cg(cp_A, cp_b, cp_x0, maxiter = maxiter, tol=1e-40)[0], iterations[:13]]
		#,"cuda_cgs": [lambda A,b,maxiter: cupyx.scipy.sparse.linalg.cgs(cp_A, cp_b, cp_x0, maxiter = maxiter, tol=1e-40)[0], iterations[:13]]
		}


"""
# to run this comparison with AMGX, please install AMGX and pyamgx and uncomment L 220-293.
title = f"Metamizer and GPU based AMGX {params.data.height}x{params.data.width}"
solvers = {"Metamizer": [lambda A,b,maxiter: neural_solver(maxiter), iterations[:10]]}|\
		{config: [lambda A,b,maxiter: pyamgx_optimizer(A, b, x0, maxiter = maxiter,rtol=1e-40,config=config), iterations[:12]] for config in ["AGGREGATION_GS"]}|\
		{config: [lambda A,b,maxiter: pyamgx_optimizer(A, b, x0, maxiter = maxiter,rtol=1e-40,config=str(config)), [10,10,20,50,100,200,500]] for config in ["AMG_CLASSICAL_L1_TRUNC","AMG_CLASSICAL_L1_AGGRESSIVE_HMIS","FGMRES_CLASSICAL_AGGRESSIVE_HMIS","PBICGSTAB"]}
solvers = {"Metamizer": [lambda A,b,maxiter: neural_solver(maxiter), iterations[:10]]
		#,"AGGREGATION_GS": [lambda A,b,maxiter: pyamgx_optimizer(A, b, x0.copy(), maxiter = maxiter,rtol=1e-40,config="AGGREGATION_GS"), iterations[:12]]
		,"AMG_CLASSICAL_L1_TRUNC": [lambda A,b,maxiter: pyamgx_optimizer(A, b, x0.copy(), maxiter = maxiter,rtol=1e-40,config="AMG_CLASSICAL_L1_TRUNC"), [10,10,20,50,100,200,500]]
		,"AMG_CLASSICAL_L1_AGGRESSIVE_HMIS": [lambda A,b,maxiter: pyamgx_optimizer(A, b, x0.copy(), maxiter = maxiter,rtol=1e-40,config="AMG_CLASSICAL_L1_AGGRESSIVE_HMIS"), [10,10,20,50,100,200,500]]
		,"FGMRES_CLASSICAL_AGGRESSIVE_HMIS": [lambda A,b,maxiter: pyamgx_optimizer(A, b, x0.copy(), maxiter = maxiter,rtol=1e-40,config="FGMRES_CLASSICAL_AGGRESSIVE_HMIS"), [10,10,20,50,100,200,500]]
		,"PCG_AGGREGATION_JACOBI": [lambda A,b,maxiter: pyamgx_optimizer(A, b, x0.copy(), maxiter = maxiter,rtol=1e-40,config="PCG_AGGREGATION_JACOBI"), [10,10,20,50,100,200,500]]
		,"AMG_CLASSICAL_AGGRESSIVE_CHEB_L1_TRUNC": [lambda A,b,maxiter: pyamgx_optimizer(A, b, x0.copy(), maxiter = maxiter,rtol=1e-40,config="AMG_CLASSICAL_AGGRESSIVE_CHEB_L1_TRUNC"), [10,10,20,50,100,200,500]]
		,"PBICGSTAB": [lambda A,b,maxiter: pyamgx_optimizer(A, b, x0.copy(), maxiter = maxiter,rtol=1e-40,config="PBICGSTAB"), [10,10,20,50,100,200,500]]}
"""

"""
title = f"Metamizer and CuPy (GPU) Sparse Linear System Solvers (Jacobi Preconditioner) {params.data.height}x{params.data.width}"
solvers = {"Metamizer": [lambda A,b,maxiter: neural_solver(maxiter), iterations[:10]]
		,"cuda_gmres": [lambda A,b,maxiter: cupyx.scipy.sparse.linalg.gmres(cp_A, cp_b, cp_x0, maxiter = maxiter, tol=1e-40, M=M)[0], iterations[:16]]
		#,"cuda_lsmr": [lambda A,b,maxiter: cupyx.scipy.sparse.linalg.lsmr(cp_A, cp_b, cp_x0, maxiter = maxiter, atol=1e-40)[0], iterations[:16]] # no Preconditioner available
		,"cuda_minres": [lambda A,b,maxiter: cupyx.scipy.sparse.linalg.minres(cp_A, cp_b, cp_x0, maxiter = maxiter, tol=1e-40, M=cp_A_diag)[0], iterations[:15]] # ValueError: indefinite preconditioner (problem mit negativen werten auf der Diagonalen?)
		,"cuda_cg": [lambda A,b,maxiter: cupyx.scipy.sparse.linalg.cg(cp_A, cp_b, cp_x0, maxiter = maxiter, tol=1e-40, M=M)[0], iterations[:13]]
		#,"cuda_cgs": [lambda A,b,maxiter: cupyx.scipy.sparse.linalg.cgs(cp_A, cp_b, cp_x0, maxiter = maxiter, tol=1e-40, M=M)[0], iterations[:13]]
		}
"""

"""
title = f"Metamizer and Scipy (CPU) Sparse Linear System Solvers {params.data.height}x{params.data.width}"
solvers = {"Metamizer": [lambda A,b,maxiter: neural_solver(maxiter), iterations[:10]]
		,"cg": [lambda A,b,maxiter: la.cg(A.copy(), b.copy(), x0.copy(), maxiter = maxiter, rtol=1e-34)[0], iterations[:13]]
		#,"cgs": [lambda A,b,maxiter: la.cgs(A.copy(), b.copy(), x0.copy(), maxiter = maxiter, rtol=1e-34)[0], iterations[:14]] # weird results
		#,"bicg": [lambda A,b,maxiter: la.bicg(A.copy(), b.copy(), x0.copy(), maxiter = maxiter, rtol=1e-34)[0], iterations[:10]]# didn't work
		,"bicgstab": [lambda A,b,maxiter: la.bicgstab(A.copy(), b.copy(), x0.copy(), maxiter = maxiter, rtol=1e-40)[0], iterations[:14]]
		,"minres": [lambda A,b,maxiter: la.minres(A.copy(), b.copy(), x0.copy(), maxiter = maxiter, rtol=1e-40)[0], iterations[:14]]
		,"qmr": [lambda A,b,maxiter: la.qmr(A.copy(), b.copy(), x0.copy(), maxiter = maxiter, rtol=1e-40)[0], iterations[:12]]
		,"tfqmr": [lambda A,b,maxiter: la.tfqmr(A.copy(), b.copy(), x0.copy(), maxiter = maxiter, rtol=1e-40)[0], iterations[:13]]
		,"gcrotmk": [lambda A,b,maxiter: la.gcrotmk(A.copy(), b.copy(), x0.copy(), maxiter = maxiter, rtol=1e-40)[0], iterations[:8]]
		,"gmres": [lambda A,b,maxiter: la.gmres(A.copy(), b.copy(), x0.copy(), maxiter = maxiter, rtol=1e-40)[0], iterations[:9]]
		,"lgmres": [lambda A,b,maxiter: la.lgmres(A.copy(), b.copy(), x0.copy(), maxiter = maxiter, rtol=1e-40)[0], iterations[:10]]}
"""

"""
#A = sp.csr_matrix(A)
title = f"Metamizer and preconditioned SciPy (CPU) Sparse Linear System Solvers {params.data.height}x{params.data.width}"
solvers = {"Metamizer": [lambda A,b,maxiter: neural_solver(maxiter), iterations[:10]]
		#,"pyAMG": [lambda A,b,maxiter: pyamg_optimizer(A.copy(), b.copy(), x0.copy(), maxiter = maxiter, rtol=1e-34), iterations[:7]] # quasi genauso schnell wie scipy AMG
		,"AMG": [lambda A,b,maxiter: scipy_cg_solver(A.copy(), b.copy(), x0.copy(), maxiter = maxiter, rtol=1e-34,preconditioner="AMG"), iterations[:7]]
		#,"IC": [lambda A,b,maxiter: scipy_cg_solver(A.copy(), b.copy(), x0.copy(), maxiter = maxiter, rtol=1e-34,preconditioner="IC"), iterations[:7]] # ist quasi in einem schritt fertig braucht aber deutlich länger?
		,"ILU": [lambda A,b,maxiter: scipy_cg_solver(A.copy(), b.copy(), x0.copy(), maxiter = maxiter, rtol=1e-34,preconditioner="ILU"), iterations[:9]]} # convergiert nicht richtig?
"""

"""
title = f"Metamizer and Scipy (CPU) Optimizers {params.data.height}x{params.data.width}"
solvers = {"Metamizer": [lambda A,b,maxiter: neural_solver(maxiter), iterations[:10]]
		#,"Powell": [lambda A,b,maxiter: scipy_optimizer(maxiter,"Powell"), iterations[:10]] # way too slow
		,"Newton-CG": [lambda A,b,maxiter: scipy_optimizer(maxiter,"Newton-CG"), iterations[:7]]
		,"Nonlinear CG": [lambda A,b,maxiter: scipy_optimizer(maxiter,"CG"), iterations[:10]]
		#,"BFGS": [lambda A,b,maxiter: scipy_optimizer(maxiter,"BFGS"), iterations[:10]] # way too slow
		,"L-BFGS-B": [lambda A,b,maxiter: scipy_optimizer(maxiter,"L-BFGS-B"), iterations[:10]]}
"""


results = {solver:{"time [s]":[],"loss":[],"loss_gt":[],"iterations":[]} for solver in solvers.keys()}

# cupy spsolve
"""
print("cupy spsolve...")
start = time.time()
x_spsolve = cupyx.scipy.sparse.linalg.spsolve(cp_A, cp_b)
duration = time.time()-start
x_spsolve = x_spsolve.reshape(params.data.height,params.data.width)
print(f"loss: {loss(x_spsolve,dataset.bc_mask)} after {duration} s")
"""


# spsolve
"""
print("spsolve...")
start = time.time()
x_spsolve = la.spsolve(A.copy(), b.copy())
duration = time.time()-start
x_spsolve = x_spsolve.reshape(params.data.height,params.data.width)
print(f"loss: {loss(x_spsolve,dataset.bc_mask)} after {duration} s")
"""

# test different solvers
for solver in solvers.keys():
	print(f"solver: {solver}")
	for maxiter in solvers[solver][1]:
		# print(f"maxiter: {maxiter}")
		start = time.time()
		x = solvers[solver][0](A,b,maxiter)
		results[solver]["time [s]"].append(time.time()-start)
		results[solver]["iterations"].append(maxiter)
		results[solver]["loss"].append(loss(x.reshape(params.data.height,params.data.width),dataset.bc_mask))
		results[solver]["loss_gt"].append(loss_gt(x.reshape(params.data.height,params.data.width),x_gt,dataset.bc_mask))
		# print(f"loss: {results[solver]['loss'][-1]} after {results[solver]['time [s]'][-1]} s")
		print(f"maxiter: {maxiter:>5} | loss: {results[solver]['loss'][-1]:.2e} | time: {results[solver]['time [s]'][-1]:.2e} s")

# visualize performance curves (residuals vs #iterations / runtime)
dpi=200
fig = plt.figure(1,figsize=(3*800/dpi,800/dpi),dpi=dpi)
colormap = plt.cm.nipy_spectral
colors = colormap(np.linspace(0,0.9,len(solvers)))
for i,x_axis in enumerate(["time [s]","iterations"]):
	ax = fig.add_subplot(1,2,1+i)
	ax.set_prop_cycle('color',colors)
	for j,solver in enumerate(solvers.keys()):
		ax.loglog(results[solver][x_axis][1:],results[solver]["loss"][1:],zorder=(2.5 if j==0 else 1))
		#ax.loglog(results[solver][x_axis][2:],results[solver]["loss"][2:],zorder=(2.5 if j==0 else 1))
	ax.set_xlabel(x_axis)
	ax.set_ylabel("loss")
	ax.legend(solvers.keys(),loc="center right")#"upper right")#
	ax.set_axisbelow(True)
	ax.grid(True, which="major", ls="-", color='0.85')
	ax.grid(True, which="minor", ls="--", color='0.95')

plt.suptitle(title)
#plt.savefig(f"plots/{title}.pdf",dpi=dpi,bbox_inches="tight")
plt.show()


"""
# visualize performance curves (residuals vs #iterations / runtime) separately in pdf format
dpi=200
for i,x_axis in enumerate(["time [s]","iterations"]):
	#plt.figure(1,figsize=(2*800/dpi,800/dpi),dpi=dpi)
	#plt.clf()
	fig, ax = plt.subplots(1,1,figsize=(2*600/dpi,700/dpi),dpi=dpi, layout='constrained')
	colormap = plt.cm.nipy_spectral
	colors = colormap(np.linspace(0,0.9,len(solvers)))
	ax.set_prop_cycle('color',colors)
	for j,solver in enumerate(solvers.keys()):
		ax.loglog(results[solver][x_axis][1:],results[solver]["loss"][1:],zorder=(2.5 if j==0 else 1))
	ax.set_xlabel(x_axis)
	ax.set_ylabel("loss")
	ax.legend(solvers.keys(),loc="center right")#"upper right")#
	ax.set_axisbelow(True)
	ax.grid(True, which="major", ls="-", color='0.85')
	ax.grid(True, which="minor", ls="--", color='0.95')
	
	plt.title(title)
	plt.savefig(f"plots/{title}_{x_axis}.pdf",dpi=dpi,bbox_inches="tight")
	plt.show()
"""



"""
# visualize difference to analytical solution
fig = plt.figure(1,figsize=(3*800/dpi,800/dpi),dpi=dpi)
plt.clf()
colormap = plt.cm.nipy_spectral
colors = colormap(np.linspace(0,0.9,len(solvers)))
for i,x_axis in enumerate(["time [s]","iterations"]):
	ax = fig.add_subplot(1,2,1+i)
	ax.set_prop_cycle('color',colors)
	for j,solver in enumerate(solvers.keys()):
		ax.loglog(results[solver][x_axis][1:],results[solver]["loss"][1:],zorder=(2.5 if j==0 else 1))
		#ax.loglog(results[solver][x_axis][2:],results[solver]["loss"][2:],zorder=(2.5 if j==0 else 1))
	ax.set_xlabel(x_axis)
	ax.set_ylabel("MSE loss with analytical solution")
	ax.legend(solvers.keys(),loc="center right")#"upper right")#

plt.suptitle(title)
plt.savefig(f"plots/{title}_MSE.png",dpi=dpi)
plt.show()
"""

# Remark: Performance depends on local hardware (GPU / CPU) => Runtimes might be different from our results!
# Our hardware:
# CPU: AMD Ryzen 9 7950X 16-Core Processor
# GPU: Nvidia GeForce RTX 4090

