# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 17:27:01 2022

@author: unknown
"""

import torch
from torch.nn import LogSoftmax, Module, Softmax, CosineSimilarity


class StandardCrossEntropy(Module):
    log_softmax = LogSoftmax()

    def __init__(self):
        super().__init__()

    def forward(self, outputs, y):
        log_probabilities = self.log_softmax(outputs)
        return -log_probabilities.gather(1, y.unsqueeze(1)).sum()/y.size(0)
    
 
class FidelityConstraint(Module):
    log_softmax = LogSoftmax()
    softmax = Softmax()
    cosim = CosineSimilarity(dim=-1)
    
    def __init__(self, cweight = 1., min_dist = 0.1):
        super().__init__()
        self.alpha = cweight
        self.thr = min_dist
    
    def forward(self, outputs, outputs0, y):
        dist = torch.abs(self.cosim(outputs, outputs0))
        xloss = torch.max(torch.as_tensor(0).cuda(), dist.sum()/y.size(0)-torch.as_tensor(self.thr).cuda())
        log_probabilities = self.log_softmax(outputs)
        celoss = -log_probabilities.gather(1, y.unsqueeze(1)).sum()/y.size(0) 
        return celoss + self.alpha*xloss
    

class LocalityConstraint(Module):
    log_softmax = LogSoftmax()
    
    def __init__(self, cweight = 1., min_grad = 0.01):
        super().__init__()
        self.alpha = cweight
        self.smt = min_grad
    
    def forward(self, outputs, grad, x, y):
        xloss = -( x*torch.log(grad+self.smt) + (torch.as_tensor(1.)-x)*torch.log(torch.as_tensor(1.)-grad+self.smt)).mean()
        log_probabilities = self.log_softmax(outputs)
        celoss = -log_probabilities.gather(1, y.unsqueeze(1)).sum()/y.size(0)
        print("locality xloss = ", xloss)
        return celoss + self.alpha*xloss


