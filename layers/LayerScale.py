import torch
import torch.nn as nn

class PoolFormerBlock(nn.Module):

    def __init__(self, dim, layer_scale_init_value=1e-5):

        super().__init__()

        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)


    def forward(self, x):

        x = x + self.drop_path( self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
            * self.token_mixer(self.norm1(x)))

        return x