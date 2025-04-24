# from baseline.meshgraphnets import cloth_model as cloth_model
# from baseline.meshgraphnets.cloth_model import Model as MeshGraphNet

import sys, os
sys.path.append(os.path.join(os.getcwd(), "../baseline/meshgraphnets"))
from impl_1 import cloth_model
from impl_1.cloth_model import Model as MeshGraphNet
from get_param2 import params
from neuralop.layers.resample import resample
from utils import normalize_grads, normalize_grads_scale, has_nan
from get_param2 import toDType
from unet_parts import *
from neuralop.models import GINO, FNO2d, FNO, UNO

device = 'cuda' if params.opt.cuda else 'cpu'

eps = 1e-12  # 1e-6#


def get_Net(params):
    if params.net == "Metamizer":
        net = Metamizer(params.hidden_size)
    return net


def get_Net2(params):
    if params.net.name == "Metamizer":
        net = toDType(Metamizer(params.net.hidden_size))
    elif params.net.name == 'FNOVertex':
        net = FNO_vertex(**params.net)
    elif params.net.name == 'UNOVertex':
        net = UNO_vertex(**params.net)
    elif params.net.name == 'MeshGraphNets':
        params.params.model = cloth_model
        net = MeshGraphNet(params.params, **params.net)
    return net


class MixedUnet(nn.Module):
    # U-Net that outputs scalar as well as field values

    def __init__(self, in_channels, out_channels, out_scalar_channels, hidden_size=64, bilinear=True):
        super(MixedUnet, self).__init__()
        self.hidden_size = hidden_size
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
        self.inc = DoubleConv(in_channels, hidden_size)
        self.down1 = Down(hidden_size, 2 * hidden_size)
        self.down2 = Down(2 * hidden_size, 4 * hidden_size)
        self.down3 = Down(4 * hidden_size, 8 * hidden_size)
        self.down4 = Down(8 * hidden_size, 16 * hidden_size // factor)
        self.up1 = Up(16 * hidden_size, 8 * hidden_size // factor, bilinear)
        self.up2 = Up(8 * hidden_size, 4 * hidden_size // factor, bilinear)
        self.up3 = Up(4 * hidden_size, 2 * hidden_size // factor, bilinear)
        self.up4 = Up(2 * hidden_size, hidden_size, bilinear)
        self.outc = OutConv(hidden_size, out_channels)
        self.out_scalar = nn.Linear(16 * hidden_size // factor, out_scalar_channels)  # TODO

    def forward(self, inputs):
        x = inputs
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        # print(f"x5 shape: {x5.shape}")
        # x_scalar = self.out_scalar(torch.mean(x5,dim=[2,3]))
        x_scalar = self.out_scalar(torch.amax(x5, dim=[2, 3]))
        return x, x_scalar


class Metamizer(nn.Module):
    # same as Grad_net_scale_inv_1_channel3, but with normalization of last step

    def __init__(self, hidden_size=64, bilinear=True):
        super(Metamizer, self).__init__()
        self.initial_scale = 0.05  # 1#0.05 # ?
        self.nn = MixedUnet(3 * 1, 1, 1, hidden_size, bilinear)

    def forward(self, grads, hidden_states=None):
        """
        :grads: input gradients of shape: batch_size x 1 x h x w
        :hidden_states: list of length batch_size
        :return:
            :step: update step of shape: batch_size x 1 x h x w
            :new_hidden_states: list of length batch_size
        """
        bs, c, h, w = grads.shape
        eps = 1e-40

        # hidden states for last gradients / last update step / scale
        hidden_states = [[torch.zeros_like(grads[0:1]), torch.zeros_like(grads[0:1]),
                          torch.ones(1, 1, 1, 1, device=device) * self.initial_scale] if hs is None else hs for hs in
                         hidden_states]

        last_grads = torch.cat([hs[0] for hs in hidden_states], 0)
        last_steps = torch.cat([hs[1] for hs in hidden_states], 0)
        last_scales = torch.cat([hs[2] for hs in hidden_states], 0)

        # normalize gradients to achieve scale invariance wrt gradients
        grad_std = grads.norm(p=2.0, dim=[1, 2, 3], keepdim=True).detach().clamp_min(eps) / np.sqrt(h * w)
        normalized_grads = 10 * torch.tanh(grads / grad_std / 10)  # "soft" tanh clamping
        normalized_last_grads = 10 * torch.tanh(last_grads / grad_std / 10)

        # normalize last update step to achieve scale invariance wrt update step size
        step_std = last_steps.norm(p=2.0, dim=[1, 2, 3], keepdim=True).detach().clamp_min(eps) / np.sqrt(h * w)
        normalized_last_steps = 10 * torch.tanh(last_steps / step_std / 10)

        # concat inputs and convert to float (half precision works as well but performance difference did not really pay off in our experiments)
        inputs = torch.cat([normalized_grads.float(), normalized_last_grads.float(), normalized_last_steps.float()], 1)

        # compute update step and delate for
        update_step, d_scale = self.nn(inputs)      # input: (bs, 3, h, w), output: (bs, 1, h, w) and (bs, 1)

        # gradient normalization (normalize_grads) => so gradients at different optimization stages get equal weights

        # convert update_step and d_scale back to double
        update_step = toDType(update_step)
        d_scale = toDType(d_scale)

        # normalize gradients
        update_step = normalize_grads(torch.tanh(update_step))

        d_scale = torch.exp(normalize_grads(2 * torch.tanh(d_scale / 2) - 1))

        # update scaling parameter
        scales = last_scales * d_scale.unsqueeze(2).unsqueeze(3)

        # multiply update step with scaling
        step = update_step * scales

        # update hidden states with new gradients / update step / scale parameters
        new_hidden_states = [[grads[i:(i + 1)].detach(), step[i:(i + 1)].detach(), scales[i:(i + 1)].detach()] for i, _
                             in enumerate(hidden_states)]

        return step, new_hidden_states

    def float(self):
        print("set model to float")
        return super().float()

    def double(self):
        print("do not set model to double")

    # return super().double()

    def type(self, dtype):
        print("set model to float")
        # return super().half() # doesn't give a lot of performance gains...
        return super().float()


class FNO_vertex(nn.Module):

    def __init__(
            self,
            *,
            in_channels=12,
            hidden_channels=64,
            out_channels=3,
            lifting_channels=256,
            projection_channels=256,
            n_layers=4,
            positional_embedding="grid",  # !NOTE: grid boundary is fixed [0, 1] x [0, 1] (synthetic data)
            n_modes_height=16,
            n_modes_width=16,
            domain_padding=(0.1, 0.1),  # percentage of padding, used for non-periodic domain
            domain_padding_mode="symmetric",  # pad both sides, can try "one-sided"
            device=None,
            dtype=None,
            shift='center',  # center, corner, None
            **kwargs
    ):
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.n_layers = n_layers
        self.positional_embedding = positional_embedding
        self.n_modes_height = n_modes_height
        self.n_modes_width = n_modes_width
        self.domain_padding = domain_padding
        self.domain_padding_mode = domain_padding_mode
        self.device = device
        self.dtype = dtype
        self.shift = shift
        self.initial_scale = 0.05

        super().__init__()
        self.model = FNO2dCustom(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
            positional_embedding=positional_embedding,
            n_modes_height=n_modes_height,
            n_modes_width=n_modes_width,
            domain_padding=domain_padding,
            domain_padding_mode=domain_padding_mode,
            device=device,
            **kwargs
        )

    def forward(self, grads, hidden_states=None):
        bs, c, h, w = grads.shape
        device = grads.device

        # hidden states for last gradients / last update step / scale
        hidden_states = [[torch.zeros_like(grads[0:1]), torch.zeros_like(grads[0:1]),
                          torch.ones(1, 1, 1, 1, device=device) * self.initial_scale] if hs is None else hs for hs in
                         hidden_states]

        last_grads = torch.cat([hs[0] for hs in hidden_states], 0)
        last_steps = torch.cat([hs[1] for hs in hidden_states], 0)
        last_scales = torch.cat([hs[2] for hs in hidden_states], 0)

        # normalize gradients to achieve scale invariance wrt gradients
        grad_std = grads.norm(p=2.0, dim=[1, 2, 3], keepdim=True).detach().clamp_min(eps) / np.sqrt(h * w)
        normalized_grads = 10 * torch.tanh(grads / grad_std / 10)  # "soft" tanh clamping
        normalized_last_grads = 10 * torch.tanh(last_grads / grad_std / 10)

        # normalize last update step to achieve scale invariance wrt update step size
        step_std = last_steps.norm(p=2.0, dim=[1, 2, 3], keepdim=True).detach().clamp_min(eps) / np.sqrt(h * w)
        normalized_last_steps = 10 * torch.tanh(last_steps / step_std / 10)

        # concat inputs and convert to float (half precision works as well but performance difference did not really pay off in our experiments)
        inputs = torch.cat([normalized_grads.float(), normalized_last_grads.float(), normalized_last_steps.float()], 1)
        # inputs = torch.cat([normalized_grads, normalized_last_grads, normalized_last_steps], 1)     # TODO precision issue

        update_step, d_scale = self.model(inputs)   # should output (bs, 1, h, w) and (bs, 1)
        # convert update_step and d_scale back to double
        update_step = toDType(update_step)
        d_scale = toDType(d_scale)

        # normalize gradients
        update_step = normalize_grads(torch.tanh(update_step))

        d_scale = torch.exp(normalize_grads(2 * torch.tanh(d_scale / 2) - 1))

        # update scaling parameter
        scales = last_scales * d_scale.unsqueeze(2).unsqueeze(3)

        # multiply update step with scaling
        step = update_step * scales

        # update hidden states with new gradients / update step / scale parameters
        new_hidden_states = [[grads[i:(i + 1)].detach(), step[i:(i + 1)].detach(), scales[i:(i + 1)].detach()] for i, _
                             in enumerate(hidden_states)]

        return step, new_hidden_states

class UNO_vertex(nn.Module):
    def __init__(
            self,
            *,
            in_channels=12,
            hidden_channels=64,
            out_channels=3,
            lifting_channels=256,
            projection_channels=256,
            uno_out_channels=[32, 64, 64, 64, 32],
            uno_n_modes=[[5, 5], [5, 5], [5, 5], [5, 5], [5, 5]],
            uno_scalings=[[1.0, 1.0], [0.5, 0.5], [1, 1], [1, 1], [2, 2]],
            horizontal_skips_map=None,
            n_layers=5,
            positional_embedding="grid",  # !NOTE: grid boundary is fixed [0, 1] x [0, 1] (synthetic data)
            domain_padding=0.2,  # percentage of padding, used for non-periodic domain
            device=None,
            dtype=None,
            shift='center',  # center, corner, None
            **kwargs
    ):
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.uno_out_channels = uno_out_channels
        self.horizontal_skips_map = horizontal_skips_map
        self.n_layers = n_layers
        self.positional_embedding = positional_embedding
        self.domain_padding = domain_padding
        self.device = device
        self.dtype = dtype
        self.shift = shift
        self.initial_scale = 0.05

        super().__init__()

        self.model = UNO2dCustom(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
            positional_embedding=positional_embedding,
            uno_n_modes=uno_n_modes,
            uno_scalings=uno_scalings,
            uno_out_channels=uno_out_channels,
            horizontal_skips_map=horizontal_skips_map,
            domain_padding=domain_padding,
            device=device,
            **kwargs
        )
        pass

    def forward(self, grads, hidden_states=None):
        bs, c, h, w = grads.shape
        device = grads.device

        # hidden states for last gradients / last update step / scale
        hidden_states = [[torch.zeros_like(grads[0:1]), torch.zeros_like(grads[0:1]),
                          torch.ones(1, 1, 1, 1, device=device) * self.initial_scale] if hs is None else hs for hs in
                         hidden_states]

        last_grads = torch.cat([hs[0] for hs in hidden_states], 0)
        last_steps = torch.cat([hs[1] for hs in hidden_states], 0)
        last_scales = torch.cat([hs[2] for hs in hidden_states], 0)

        # normalize gradients to achieve scale invariance wrt gradients
        grad_std = grads.norm(p=2.0, dim=[1, 2, 3], keepdim=True).detach().clamp_min(eps) / np.sqrt(h * w)
        normalized_grads = 10 * torch.tanh(grads / grad_std / 10)  # "soft" tanh clamping
        normalized_last_grads = 10 * torch.tanh(last_grads / grad_std / 10)

        # normalize last update step to achieve scale invariance wrt update step size
        step_std = last_steps.norm(p=2.0, dim=[1, 2, 3], keepdim=True).detach().clamp_min(eps) / np.sqrt(h * w)
        normalized_last_steps = 10 * torch.tanh(last_steps / step_std / 10)

        # concat inputs and convert to float (half precision works as well but performance difference did not really pay off in our experiments)
        inputs = torch.cat([normalized_grads.float(), normalized_last_grads.float(), normalized_last_steps.float()], 1)

        update_step, d_scale = self.model(inputs)   # should output (bs, 1, h, w) and (bs, 1)
        # convert update_step and d_scale back to double
        update_step = toDType(update_step)
        d_scale = toDType(d_scale)

        # normalize gradients
        update_step = normalize_grads(torch.tanh(update_step))

        d_scale = torch.exp(normalize_grads(2 * torch.tanh(d_scale / 2) - 1))

        # update scaling parameter
        scales = last_scales * d_scale.unsqueeze(2).unsqueeze(3)

        # multiply update step with scaling
        step = update_step * scales

        # update hidden states with new gradients / update step / scale parameters
        new_hidden_states = [[grads[i:(i + 1)].detach(), step[i:(i + 1)].detach(), scales[i:(i + 1)].detach()] for i, _
                             in enumerate(hidden_states)]

        return step, new_hidden_states


class FNO2dCustom(FNO2d):
    def __init__(
        self,
        hidden_channels,
        **kwargs
    ):
        super().__init__(
            hidden_channels=hidden_channels,
            **kwargs
        )
        self.out_scalar = nn.Linear(hidden_channels, 1)
        pass

    def forward(self, x, output_shape=None, **kwargs):

        if output_shape is None:
            output_shape = [None] * self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None] * (self.n_layers - 1) + [output_shape]

        # append spatial pos embedding if set
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        scale = self.out_scalar(torch.amax(x, dim=[2, 3]))

        out = self.projection(x)

        return out, scale


class UNO2dCustom(UNO):
    def __init__(
        self,
        hidden_channels,
        scale_feature_layer,
        uno_out_channels,
        **kwargs
    ):
        super().__init__(
            hidden_channels=hidden_channels,
            scale_feature_layer=scale_feature_layer,
            uno_out_channels=uno_out_channels,
            **kwargs
        )
        self.scale_feature_layer = scale_feature_layer
        self.uno_out_channels = uno_out_channels
        self.out_scalar = nn.Linear(uno_out_channels[scale_feature_layer], 1)
        pass

    def forward(self, x, output_shape=None, **kwargs):

        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)
        output_shape = [
            int(round(i * j))
            for (i, j) in zip(x.shape[-self.n_dim:], self.end_to_end_scaling_factor)
        ]

        skip_outputs = {}
        cur_output = None
        for layer_idx in range(self.n_layers):
            if layer_idx in self.horizontal_skips_map.keys():
                skip_val = skip_outputs[self.horizontal_skips_map[layer_idx]]
                output_scaling_factors = [
                    m / n for (m, n) in zip(x.shape, skip_val.shape)
                ]
                output_scaling_factors = output_scaling_factors[-1 * self.n_dim:]
                t = resample(
                    skip_val, output_scaling_factors, list(range(-self.n_dim, 0))
                )
                x = torch.cat([x, t], dim=1)

            if layer_idx == self.n_layers - 1:
                cur_output = output_shape
            x = self.fno_blocks[layer_idx](x, output_shape=cur_output)
            # (36, 32, 100, 100), (36, 64, 50, 50), (36, 64, 50, 50), (36, 64, 50, 50), (36, 32, 100, 100)

            if layer_idx == self.scale_feature_layer:
                scale_feature = x

            if layer_idx in self.horizontal_skips_map.values():
                skip_outputs[layer_idx] = self.horizontal_skips[str(layer_idx)](x)

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        scale = self.out_scalar(torch.amax(scale_feature, dim=[2, 3]))

        x = self.projection(x)
        return x, scale