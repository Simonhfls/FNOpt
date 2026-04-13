from utils import normalize_grads
from get_param import toDType
from unet_parts import *
from neuralop.models import FNO2d
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_Net(params):
    if params.net.name == "Metamizer":
        basenet = MixedUnet(3 * 1, 1, 1, params.net.hidden_size, bilinear=True)
        net = toDType(FNOptUnified(basenet))
    elif params.net.name == 'FNOVertex':
        basenet = FNOpt(**params.net)
        net = FNOptUnified(basenet)
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
        x5 = self.down4(x4)     # (bs, 160, h/16, w/16)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        x_scalar = self.out_scalar(torch.amax(x5, dim=[2, 3]))
        return x, x_scalar


class FNOpt(FNO2d):
    def __init__(
        self,
        hidden_channels,
        scale_feature_layer=None,
        scalar_pooling='amax',
        **kwargs
    ):
        super().__init__(
            hidden_channels=hidden_channels,
            **kwargs
        )
        self.out_scalar = nn.Linear(hidden_channels, 1)
        self.scale_feature_layer = scale_feature_layer
        self.scalar_pooling = scalar_pooling  # 'amax' or 'avgpool'

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

            if self.scale_feature_layer is not None and layer_idx == self.scale_feature_layer:
                scale_feature = x


        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        if self.scale_feature_layer is None:
            scale_feature = x


        if self.scalar_pooling == 'amax':
            scale_feature = torch.amax(scale_feature, dim=[2, 3])
        elif self.scalar_pooling == 'avgpool':
            scale_feature = F.adaptive_avg_pool2d(scale_feature, 1).view(scale_feature.size(0), -1)
        else:
            raise ValueError(f"Unsupported scalar_pooling mode: {self.scalar_pooling}")
        scale = self.out_scalar(scale_feature)
        out = self.projection(x)

        return out, scale


class FNOptUnified(nn.Module):
    # Code adapted from Metamizer

    def __init__(self, basenet):
        super(FNOptUnified, self).__init__()
        self.initial_scale = 0.05
        self.nn = basenet
    def forward(self, grads, hidden_states=None, **kwargs):
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
        # update_step, d_scale = self.nn(inputs)      # input: (bs, 3, h, w), output: (bs, 1, h, w) and (bs, 1)
        if "output_shape" in kwargs:
            update_step, d_scale = self.nn(inputs, output_shape=kwargs["output_shape"])
        else:
            update_step, d_scale = self.nn(inputs)

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

    def type(self, dtype):
        print("set model to float")
        return super().float()

