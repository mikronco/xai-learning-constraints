# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 15:25:02 2022

@author: unknown
"""
import sys
import numpy as np
import torch
from torch.autograd import Variable
from utils.utils import input_grads
from losses.losses import StandardCrossEntropy, FidelityConstraint, LocalityConstraint, GradientRegularization, ConsistencyConstraint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_base(model, loaders, num_epochs, optimizer, loss_func = StandardCrossEntropy()): 
    model.train()
    total_step = len(loaders['train'])
    acc_x_epoch = []
    loss_x_batch = [] 
    for epoch in range(num_epochs):
        correct = 0
        for i, (images, labels) in enumerate(loaders['train']):
            # gives batch data, normalize x when iterate train_loader
            images, labels = images.to(device), labels.to(device)
            b_x = Variable(images)   # batch x
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



def train_fidelity(model, loaders, num_epochs, optimizer, loss_func = FidelityConstraint()):
    model.train()        
    total_step = len(loaders['train'])
    acc_x_epoch = []
    loss_x_batch = []
        
    for epoch in range(num_epochs):
        correct = 0
        for i, (images, labels) in enumerate(loaders['train']):
            # gives batch data, normalize x when iterate train_loader
            images, labels = images.to(device), labels.to(device)
            b_x = Variable(images, requires_grad = True)   # batch x
            b_y = Variable(labels)   # batch y
            output = model(b_x)
            grads = input_grads(output, b_x, b_y)     
            gmax = grads.view(b_x.size(0), 1, -1).max(2).values.view(b_x.size(0), 1, 1, 1)
            gmin = grads.view(b_x.size(0), 1, -1).min(2).values.view(b_x.size(0), 1, 1, 1)
            ngrad = (grads - gmin)/(gmax - gmin)
            b_x_masked = b_x*(1-ngrad)
            output_masked = model(b_x_masked)
            loss = loss_func(output, output_masked, b_y)
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




def train_locality(model, loaders, num_epochs, optimizer, loss_func = LocalityConstraint()): 
    model.train()        
    total_step = len(loaders['train'])    
    acc_x_epoch = []
    loss_x_batch = []
        
    for epoch in range(num_epochs):
        correct = 0
        for i, (images, labels) in enumerate(loaders['train']):
            # gives batch data, normalize x when iterate train_loader
            images, labels = images.to(device), labels.to(device)
            b_x = Variable(images, requires_grad = True)   # batch x
            b_y = Variable(labels)   # batch y
            output = model(b_x)
            grads = input_grads(output, b_x, b_y)     
            gmax = grads.view(b_x.size(0), 1, -1).max(2).values.view(b_x.size(0), 1, 1, 1)
            gmin = grads.view(b_x.size(0), 1, -1).min(2).values.view(b_x.size(0), 1, 1, 1)
            ngrad = (grads - gmin)/(gmax - gmin)
            loss = loss_func(output, ngrad, b_x, b_y)
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


def train_gradreg(model, loaders, num_epochs, optimizer, constraint = "grad_reg", alpha = 1.): 
    model.train()        
    total_step = len(loaders['train'])    
    acc_x_epoch = []
    loss_x_batch = []
    
    if constraint == "grad_reg":
        loss_func = GradientRegularization()
    elif constraint == "consistency":
        loss_func = ConsistencyConstraint(cweight = alpha)
    else:
        sys.exit("Specify valid penalty term!")
    
    for epoch in range(num_epochs):
        correct = 0
        for i, (images, labels) in enumerate(loaders['train']):
            # gives batch data, normalize x when iterate train_loader
            images, labels = images.to(device), labels.to(device)
            b_x = Variable(images, requires_grad = True)   # batch x
            b_y = Variable(labels)   # batch y
            output = model(b_x)
            grads = input_grads(output, b_x, b_y)     
            loss = loss_func(output, grads, b_y)
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

