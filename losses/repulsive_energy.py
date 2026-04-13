import torch
from torch import nn

class RepulsiveEnergy(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
        self.name = 'repulsive_energy'

    def calc_single(self, pred_pos, f_connectivity_edges):
        """
        pred_pos: (N, 3)
        f_connectivity_edges: (E, 2)
        """

        dist = torch.square(pred_pos[None] - pred_pos[:, None]).sum(-1) + 1e-8
        mask = torch.ones_like(dist)
        mask[dist > self.threshold * self.threshold] = 0

        mask[f_connectivity_edges[:, 0], f_connectivity_edges[:, 1]] = 0
        mask[f_connectivity_edges[:, 1], f_connectivity_edges[:, 0]] = 0
        mask[torch.arange(mask.shape[0]), torch.arange(mask.shape[0])] = 0

        masked_dist = -torch.log(dist) * mask
        loss = torch.sum(masked_dist)

        return loss

    # @torch.jit.script_method
    def forward(self, pred_pos, f_connectivity_edges):
        """
        pred_pos: (B, N, 3)
        f_connectivity_edges: (E, 2)
        """
        loss_list = []
        B = pred_pos.shape[0]
        for i in range(B):
            loss_sample = self.calc_single(pred_pos[i], f_connectivity_edges)
            loss_list.append(loss_sample)

        loss = sum(loss_list) / B

        return loss


