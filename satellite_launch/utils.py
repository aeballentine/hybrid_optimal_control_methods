from torch import nn
from collections import OrderedDict
import torch
import pandas as pd
import os


class NeuralNetworkSimple(nn.Module):
    def __init__(self, dimensions, function=None, output_activation=None):
        super().__init__()

        stacking = OrderedDict()
        i = 0
        while len(dimensions) > 2:
            stacking["lin" + str(i)] = nn.Linear(dimensions[0], dimensions[1])
            if function:
                stacking["act" + str(i)] = function
            dimensions.pop(0)
            i += 1
        stacking["lin" + str(i)] = nn.Linear(dimensions[0], dimensions[1])
        if output_activation is not None:
            stacking["act" + str(i)] = output_activation
        self.linear_stack = nn.Sequential(stacking)

    def forward(self, inp):
        # logits = self.linear_stack(torch.cat(inp, dim=1))
        logits = self.linear_stack(inp)
        # return logits.view(-1, 4, 4)
        return logits