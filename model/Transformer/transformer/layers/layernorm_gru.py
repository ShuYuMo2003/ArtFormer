import math
import torch
from torch import nn
from torch.nn import Module

class LayerNormGRUCell(nn.Module):
    '''
    This part comes from https://github.com/ElektrischesSchaf/LayerNorm_GRU/blob/main/GRU_layernorm_cell.py
    '''
    def __init__(self, input_size, hidden_size, bias=True):
        super(LayerNormGRUCell, self).__init__()

        self.ln_i2h = nn.LayerNorm(2 * hidden_size, elementwise_affine=False)
        self.ln_h2h = nn.LayerNorm(2 * hidden_size, elementwise_affine=False)
        self.ln_cell_1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln_cell_2 = nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.i2h = nn.Linear(input_size, 2 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 2 * hidden_size, bias=bias)
        self.h_hat_W = nn.Linear(input_size, hidden_size, bias=bias)
        self.h_hat_U = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.hidden_size = hidden_size
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, h):  # bs,hid
        h = h
        h = h.view(h.size(0), -1)
        x = x.view(x.size(0), -1)

        # Linear mappings
        i2h = self.i2h(x)
        h2h = self.h2h(h)

        # Layer norm
        i2h = self.ln_i2h(i2h)
        h2h = self.ln_h2h(h2h)

        preact = i2h + h2h

        # activations
        gates = preact[:, :].sigmoid()
        z_t = gates[:, :self.hidden_size]
        r_t = gates[:, -self.hidden_size:]

        # h_hat
        h_hat_first_half = self.h_hat_W(x)
        h_hat_last_half = self.h_hat_U(h)

        # layer norm
        h_hat_first_half = self.ln_cell_1(h_hat_first_half)
        h_hat_last_half = self.ln_cell_2(h_hat_last_half)

        h_hat = torch.tanh(h_hat_first_half + torch.mul(r_t, h_hat_last_half))
        h_t = torch.mul(1 - z_t, h) + torch.mul(z_t, h_hat)
        h_t = h_t.view(h_t.size(0), -1)
        return h_t