import torch
import torch.nn.functional as F
import math
from get_param2 import params, toCuda, toCpu

# def toCuda(x):
# 	if type(x) is tuple:
# 		return [xi.cuda() if params.cuda else xi for xi in x]
# 	return x.cuda() if params.cuda else x
#
# def toCpu(x):
# 	if type(x) is tuple:
# 		return [xi.detach().cpu() for xi in x]
# 	return x.detach().cpu()


# First order derivatives (d/dx)

dx_kernel = toCuda(torch.Tensor([-0.5,0,0.5]).unsqueeze(0).unsqueeze(1).unsqueeze(2))
def dx(v):
	return F.conv2d(v,dx_kernel,padding=(0,1))
def dx_periodic(v):
	return F.conv2d(F.pad(v,pad=(1,1,0,0,0,0),mode='circular'),dx_kernel)

dx_left_kernel = toCuda(torch.Tensor([-1,1,0]).unsqueeze(0).unsqueeze(1).unsqueeze(2))
def dx_left(v):
	return F.conv2d(v,dx_left_kernel,padding=(0,1))

dx_right_kernel = toCuda(torch.Tensor([0,-1,1]).unsqueeze(0).unsqueeze(1).unsqueeze(2))
def dx_right(v):
	return F.conv2d(v,dx_right_kernel,padding=(0,1))

# First order derivatives (d/dy)

dy_kernel = toCuda(torch.Tensor([-0.5,0,0.5]).unsqueeze(0).unsqueeze(1).unsqueeze(3))
def dy(v):
	return F.conv2d(v,dy_kernel,padding=(1,0))
def dy_periodic(v):
	return F.conv2d(F.pad(v,pad=(0,0,1,1,0,0),mode='circular'),dy_kernel)

dy_top_kernel = toCuda(torch.Tensor([-1,1,0]).unsqueeze(0).unsqueeze(1).unsqueeze(3))
def dy_top(v):
	return F.conv2d(v,dy_top_kernel,padding=(1,0))

dy_bottom_kernel = toCuda(torch.Tensor([0,-1,1]).unsqueeze(0).unsqueeze(1).unsqueeze(3))
def dy_bottom(v):
	return F.conv2d(v,dy_bottom_kernel,padding=(1,0))

# Curl operator

def rot_mac(a):
	return torch.cat([-dx_right(a),dy_bottom(a)],dim=1)

# Laplace operator

#laplace_kernel = toCuda(torch.Tensor([[0,1,0],[1,-4,1],[0,1,0]]).unsqueeze(0).unsqueeze(1)) # 5 point stencil
#laplace_kernel = toCuda(torch.Tensor([[1,1,1],[1,-8,1],[1,1,1]]).unsqueeze(0).unsqueeze(1)) # 9 point stencil
laplace_kernel = toCuda(0.25*torch.Tensor([[1,2,1],[2,-12,2],[1,2,1]]).unsqueeze(0).unsqueeze(1)) # isotropic 9 point stencil
def laplace(v):
	return F.conv2d(v,laplace_kernel,padding=(1,1))

laplace_detach_kernel = toCuda(0.25*torch.Tensor([[1,2,1],[2,0,2],[1,2,1]]).unsqueeze(0).unsqueeze(1))
def laplace_detach(v):  #same result as laplace but with detached gradients of neighborhood
	return F.conv2d(v.detach(),laplace_detach_kernel,padding=(1,1)) - 3*v

def laplace_periodic(v):
	return F.conv2d(F.pad(v,pad=(1,1,1,1,0,0),mode='circular'),laplace_kernel)

def laplace_periodic_detach(v):
	return F.conv2d(F.pad(v.detach(),pad=(1,1,1,1,0,0),mode='circular'),laplace_detach_kernel) - 3*v


# mapping operators

map_vx2vy_kernel = 0.25*toCuda(torch.Tensor([[0,1,1],[0,1,1],[0,0,0]]).unsqueeze(0).unsqueeze(1))
def map_vx2vy(v):
	return F.conv2d(v,map_vx2vy_kernel,padding=(1,1))

map_vx2vy_left_kernel = 0.5*toCuda(torch.Tensor([[0,1,0],[0,1,0],[0,0,0]]).unsqueeze(0).unsqueeze(1))
def map_vx2vy_left(v):
	return F.conv2d(v,map_vx2vy_left_kernel,padding=(1,1))

map_vx2vy_right_kernel = 0.5*toCuda(torch.Tensor([[0,0,1],[0,0,1],[0,0,0]]).unsqueeze(0).unsqueeze(1))
def map_vx2vy_right(v):
	return F.conv2d(v,map_vx2vy_right_kernel,padding=(1,1))

map_vy2vx_kernel = 0.25*toCuda(torch.Tensor([[0,0,0],[1,1,0],[1,1,0]]).unsqueeze(0).unsqueeze(1))
def map_vy2vx(v):
	return F.conv2d(v,map_vy2vx_kernel,padding=(1,1))

map_vy2vx_top_kernel = 0.5*toCuda(torch.Tensor([[0,0,0],[1,1,0],[0,0,0]]).unsqueeze(0).unsqueeze(1))
def map_vy2vx_top(v):
	return F.conv2d(v,map_vy2vx_top_kernel,padding=(1,1))

map_vy2vx_bottom_kernel = 0.5*toCuda(torch.Tensor([[0,0,0],[0,0,0],[1,1,0]]).unsqueeze(0).unsqueeze(1))
def map_vy2vx_bottom(v):
	return F.conv2d(v,map_vy2vx_bottom_kernel,padding=(1,1))


mean_left_kernel = 0.5*toCuda(torch.Tensor([1,1,0]).unsqueeze(0).unsqueeze(1).unsqueeze(2))
def mean_left(v):
	return F.conv2d(v,mean_left_kernel,padding=(0,1))

mean_top_kernel = 0.5*toCuda(torch.Tensor([1,1,0]).unsqueeze(0).unsqueeze(1).unsqueeze(3))
def mean_top(v):
	return F.conv2d(v,mean_top_kernel,padding=(1,0))

mean_right_kernel = 0.5*toCuda(torch.Tensor([0,1,1]).unsqueeze(0).unsqueeze(1).unsqueeze(2))
def mean_right(v):
	return F.conv2d(v,mean_right_kernel,padding=(0,1))

mean_bottom_kernel = 0.5*toCuda(torch.Tensor([0,1,1]).unsqueeze(0).unsqueeze(1).unsqueeze(3))
def mean_bottom(v):
	return F.conv2d(v,mean_bottom_kernel,padding=(1,0))


def staggered2normal(v):
	v[:,0:1] = mean_left(v[:,0:1])
	v[:,1:2] = mean_top(v[:,1:2])
	return v

def normal2staggered(v):#CODO: double-check that! -> seems correct
	v[:,0:1] = mean_right(v[:,0:1])
	v[:,1:2] = mean_bottom(v[:,1:2])
	return v



def vector2HSV(vector,plot_sqrt=False):
	"""
	transform vector field into hsv color wheel
	:vector: vector field (size: 2 x height x width)
	:return: hsv (hue: direction of vector; saturation: 1; value: abs value of vector)
	"""
	values = torch.sqrt(torch.sum(torch.pow(vector,2),dim=0)).unsqueeze(0)
	saturation = toCuda(torch.ones(values.shape))
	norm = vector/(values+0.000001)
	angles = torch.asin(norm[0])+math.pi/2
	angles[norm[1]<0] = 2*math.pi-angles[norm[1]<0]
	hue = angles.unsqueeze(0)/(2*math.pi)
	hue = (hue*360+100)%360
	#values = norm*torch.log(values+1)
	values = values/torch.max(values)
	if plot_sqrt:
		values = torch.sqrt(values)
	hsv = torch.cat([hue,saturation,values])
	return hsv.permute(1,2,0).cpu().numpy()
