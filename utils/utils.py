# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 17:39:13 2022

@author: unknown
"""

import torch 
import numpy as np
import random


def input_grads(outputs, x, y):
    return torch.stack([torch.autograd.grad(outputs=out, inputs=x, retain_graph=True, create_graph=True)[0][i] 
                             for i, out in enumerate(outputs.gather(1, y.unsqueeze(1)))])

def integrated_grads(model, x, x_base, y, m = 10, n = 4):
    steps = list(np.linspace(1, m, num=n, dtype=int, axis=0))
    return (x-x_base)*torch.stack([input_grads(model(x_base + k*(x-x_base)/m), x, y) for k in steps]).mean(0)



class AddGaussianNoise(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size())*self.std  + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
    
class AddSquareMask(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, tensor):
        i = random.randint(self.size, tensor.size()[-1]-self.size)
        j = random.randint(self.size, tensor.size()[-1]-self.size)
        sqr = torch.zeros(tensor.size())
        sqr[i-self.size:i+self.size, j-self.size:j+self.size] = random.uniform(0,1)
        return tensor + sqr
    
    def __repr__(self):
        return self.__class__.__name__ 