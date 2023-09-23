import torch
import torch.nn.functional as F
from torch.autograd import Variable

import torch.nn as nn
import numpy as np

def gs_gs(x, a = 1 , b = 0,c = 0 ):

    return torch.exp(-(x) * (x) * a + b) + c

class pish(nn.Module):
    def __init__(self):
        '''
        Init method.
        '''

        super().__init__()

        self.beta1 = nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
        self.beta2 = nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
        self.beta3 = nn.Parameter(torch.Tensor([0.5]), requires_grad=True)

    def forward(self, input):
        '''
        Forward pass of the function.
        '''

        b1 = gs_gs(self.beta1, 4, -3.0, 0.0)
        b2 = gs_gs(self.beta2, 4, 0.0, 0.0)
        b3 = gs_gs(self.beta3, 4, b2, 100.0)


        z1 = (1.0 / (b1 + torch.exp(-input)))
        z2 = (1.0 / (b3 + torch.exp(-input)))

        return input * torch.tanh(torch.where(input > 0, z1, z2))


