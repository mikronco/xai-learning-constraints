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
    
 
class GradientRegularization(Module):
    log_softmax = LogSoftmax()

    def __init__(self, cweight = 0.1):
        super().__init__()
        self.alpha = cweight

    def forward(self, outputs, grad, y):
        log_probabilities = self.log_softmax(outputs)
        xloss = torch.abs(grad).mean()
        return -log_probabilities.gather(1, y.unsqueeze(1)).sum()/y.size(0)+self.alpha*xloss
    
 
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
        return celoss + self.alpha*xloss
    

class ConsistencyConstraint(Module):
    log_softmax = LogSoftmax()
    softmax = Softmax()
    cosim = CosineSimilarity(dim=-1)
    
    def __init__(self, cweight = 1.):
        super().__init__()
        self.alpha = cweight
    
    def forward(self, outputs, grad, y):
        xloss = 0
        gmax = grad.view(grad.size(0), 1, -1).max(2).values.view(grad.size(0), 1, 1, 1)
        gmin = grad.view(grad.size(0), 1, -1).min(2).values.view(grad.size(0), 1, 1, 1)
        ngrad = (grad - gmin)/(gmax - gmin)
        for n in range(10):
            cgrad = ngrad[(torch.argmax(self.softmax(outputs), dim=1) == n),:,:,:]
            cgrad = cgrad.reshape(cgrad.shape[0], cgrad.shape[-1]*cgrad.shape[-1])
            for i in range(cgrad.shape[0]):
                for j in range((i+1),cgrad.shape[0]):
                    xloss += 1-self.cosim(cgrad[i,:], cgrad[j,:])
        xloss /= y.size(0)
        log_probabilities = self.log_softmax(outputs)
        celoss = -log_probabilities.gather(1, y.unsqueeze(1)).sum()/y.size(0)
        return celoss + self.alpha*xloss


class GeneralizabilityConstraint(Module): 
    log_softmax = LogSoftmax()
    softmax = Softmax()
    cosim = CosineSimilarity(dim=-1)
    
    def __init__(self, cweight = 1.):
        super().__init__()
        self.alpha = cweight
    
    def forward(self, outputs, ngrad, model, x, y):
        xloss = 0
        for n in range(10):
            cgrad = ngrad[(torch.argmax(self.softmax(outputs), dim=1) == n),:,:,:]
            for i in range(cgrad.shape[0]):
                x1 = x[(torch.argmax(self.softmax(outputs), dim=1) == n),:,:,:]*cgrad[i,:,:,:].unsqueeze(0)
                outputs1 = model(x1)
                log_prob1 = self.log_softmax(outputs1)
                xloss += log_prob1.gather(1, y[(torch.argmax(self.softmax(outputs), dim=1) == n)].unsqueeze(1)).sum()/y[(torch.argmax(self.softmax(outputs), dim=1) == n)].size(0)
        log_probabilities = self.log_softmax(outputs)
        celoss = -log_probabilities.gather(1, y.unsqueeze(1)).sum()/y.size(0)
        return celoss - self.alpha*xloss

