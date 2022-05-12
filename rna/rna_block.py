from turtle import forward
from colorama import Fore
from torch import nn
import torch
from itertools import combinations as combo
import torch.nn.functional as F

class SqEx(nn.Module):

    def __init__(self, n_features, reduction=16):
        super(SqEx, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 16)')

        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):
        y = self.nonlin1(self.linear1(x))
        y = self.nonlin2(self.linear2(y))
        y = x * y
        return y

class RNABlock(nn.Module):
    def __init__(self, modalities, input_dim, output_dim, net_type: str, net_dropout: float):
        super().__init__()
        self.input_dim = input_dim
        if net_type.lower() == 'sqex':
            self.fc = nn.ModuleDict({m:
            nn.Sequential(
                SqEx(input_dim),
                nn.Dropout(p=net_dropout)
            ) 
            for m in modalities})
        elif net_type.lower() == 'linear':
            print(Fore.GREEN + 'Output dim of additional net linear layer: ', output_dim)
            self.fc = nn.ModuleDict({m: nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(p=net_dropout)
                ) for m in modalities})
        else:
            raise NotImplementedError()

    def forward(self, input):
        # Input is expected to be a tensor of dim (bs * num_segments, input_dim * num_modalities)
        input_chuncks = input.split(self.input_dim, dim=-1) # to [(bs * num_segments, input_dim)] x num_modalities
        output_chunks = [fc(x) for fc,x in zip(self.fc.values(), input_chuncks)]
        return torch.cat(output_chunks, dim=-1), [c.norm(p=2, dim=-1).mean() for c in output_chunks]


class RNALoss(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
    def forward(self, output):
        # Output is expected to be a tensor of dim (bs * num_segments, input_dim * num_modalities)
        output_chuncks = output.split(self.input_dim, dim=-1) # to [(bs * num_segments, input_dim)] x num_modalities
        # Align inter-modality norms
        mfn = [out.norm(p=2, dim=-1).mean() for out in output_chuncks]
        mfn.sort() # ascending order
        loss = 0
        for f1, f2 in combo(mfn, 2):
            loss += (f1/f2 - 1)**2
        return loss

class SAFNLoss(nn.Module):
    def __init__(self, input_dim: int, delta: float) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.delta = delta
    def forward(self, input):
        # For each modality separately
        input_chuncks = input.split(self.input_dim, dim=-1) # to [(bs * num_segments, input_dim)] x num_modalities
        loss = 0
        for inp in input_chuncks:
            radius = inp.norm(p=2, dim=-1).detach()
            radius = radius + self.delta
            loss += ((inp.norm(p=2, dim=-1) - radius) ** 2).mean()
        return loss
