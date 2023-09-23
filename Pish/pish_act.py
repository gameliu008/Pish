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

        self.p1 = nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
        self.p2 = nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
        self.p3 = nn.Parameter(torch.Tensor([0.5]), requires_grad=True)

    def forward(self, x):
        '''
        Forward pass of the function.
        '''

        a1 = gs_gs(self.p1, 4, 0.0, 0.0)
        a2 = gs_gs(self.p2, 4, 0.0, 0.0)
        a3 = gs_gs(self.p3, 4, a2, 1.0)
        #a3 = gs_gs(self.p3, 4, a2, 100.0)
        b = 0.0
        z1 = (1.0 / (a1 + torch.exp(-x)))
        z2 = (1.0 / (a3 + torch.exp(-x))) + b

        return x * torch.tanh(torch.where(x > 0, z1, z2))
