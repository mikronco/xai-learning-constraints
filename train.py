# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 15:25:02 2022

@author: unknown
"""
import sys
import numpy as np
import torch
from torch.autograd import Variable
from utils.utils import input_grads, integrated_grads
from losses.losses import StandardCrossEntropy, FidelityConstraint, SmoothnessConstraint, LocalityConstraint, GradientRegularization, ConsistencyConstraint, SymmetryConstraint
from torch.nn import Softmax

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_base(model, loaders, num_epochs, optimizer):
    model.train()
    total_step = len(loaders['train'])
    acc_x_epoch = []
    loss_x_batch = [] 
    loss_func = StandardCrossEntropy()
    for epoch in range(num_epochs):
        correct = 0
        for i, (images, labels) in enumerate(loaders['train']):
            # gives batch data, normalize x when iterate train_loader
            images, labels = images.to(device), labels.to(device)
            b_x = Variable(images)    # batch x
            b_y = Variable(labels)   # batch y
            output = model(b_x)
            loss = loss_func(output, b_y)
            loss_x_batch.append(loss)
            flat_out = np.argmax(output.detach().cpu().numpy(), axis=1)
            correct += (flat_out == b_y.detach().cpu().numpy()).sum()
            optimizer.zero_grad()           
            loss.backward()    
            optimizer.step()                
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                pass
            
            if (i + 1) == total_step:
                accuracy =  correct / (total_step*loaders['train'].batch_size)
                print('Accuracy = ', accuracy)
        
        acc_x_epoch.append(accuracy)
    
    return (acc_x_epoch, loss_x_batch)



def train_xai(model, loaders, num_epochs, optimizer, penalty = "base", attribution = "gradients", alpha = 0.1):
    constraints = {"grad_reg": GradientRegularization(cweight = alpha), 
               "consistency": ConsistencyConstraint(cweight = alpha), 
               "smoothness": SmoothnessConstraint(cweight = alpha), 
               "fidelity": FidelityConstraint(cweight = alpha), 
               "locality": LocalityConstraint(cweight = alpha), 
               "symmetry": SymmetryConstraint(cweight=alpha)}
    model.train()
    total_step = len(loaders['train'])
    acc_x_epoch = []
    loss_x_batch = [] 
    loss_func = constraints[penalty]
    for epoch in range(num_epochs):
        correct = 0
        for i, (images, labels) in enumerate(loaders['train']):
            # gives batch data, normalize x when iterate train_loader
            images, labels = images.to(device), labels.to(device)
            b_x = Variable(images, requires_grad = True)    # batch x
            b_y = Variable(labels)   # batch y
            output = model(b_x)
            grads = input_grads(output, b_x, b_y)
            loss = loss_func(output, grads, b_x, model, b_y)
            loss_x_batch.append(loss)
            flat_out = np.argmax(output.detach().cpu().numpy(), axis=1)
            correct += (flat_out == b_y.detach().cpu().numpy()).sum()
            optimizer.zero_grad()           
            loss.backward()    
            optimizer.step()                
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                pass
            
            if (i + 1) == total_step:
                accuracy =  correct / (total_step*loaders['train'].batch_size)
                print('Accuracy = ', accuracy)
        
        acc_x_epoch.append(accuracy)
    
    return (acc_x_epoch, loss_x_batch)

