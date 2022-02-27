# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 17:39:13 2022

@author: unknown
"""

import torch 


def input_grads(outputs, x, y):
    return torch.stack([torch.autograd.grad(outputs=out, inputs=x, retain_graph=True, create_graph=True)[0][i] 
                             for i, out in enumerate(outputs.gather(1, y.unsqueeze(1)))])