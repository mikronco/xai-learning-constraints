# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 17:39:13 2022

@author: unknown
"""

import torch 
import numpy as np


def input_grads(outputs, x, y):
    return torch.stack([torch.autograd.grad(outputs=out, inputs=x, retain_graph=True, create_graph=True)[0][i] 
                             for i, out in enumerate(outputs.gather(1, y.unsqueeze(1)))])

def integrated_grads(model, x, x_base, y, m = 10, n = 4):
    steps = list(np.linspace(1, m, num=n, dtype=int, axis=0))
    return (x-x_base)*torch.stack([input_grads(model(x_base + k*(x-x_base)/m), x, y) for k in steps]).mean(0)