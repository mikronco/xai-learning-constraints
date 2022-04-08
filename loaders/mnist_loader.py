# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 16:40:16 2022

@author: unknown
"""

import torch 
from torchvision import datasets
from torchvision.transforms import ToTensor
from captum.attr import Saliency
import numpy as np

device = torch.device('cpu')

def MNIST_data(batch_size = 20, test_batch_size = 1, roar = False, model = None, perc = None):
    
    train_data = datasets.MNIST(
            root = 'data',
            train = True,                         
            transform = ToTensor(), 
            download = True,            
        )
    test_data = datasets.MNIST(
        root = 'data', 
        train = False, 
        transform = ToTensor()
    )
    
    if roar:
        model.to(device)
        grad = Saliency(model)
        attr = grad.attribute(train_data.data.float().unsqueeze(1), target=train_data.targets)
        q = [np.percentile(at[at!=0].flatten().numpy(), perc) for at in attr[:,0,:,:]]
        maskpos = torch.ones((attr.shape))
        for i in range(attr.shape[0]):
            maskpos[i,0,attr[i,0,:,:]>q[i]]=0
            train_data.data[i,:,:] = train_data.data[i,:,:]*maskpos[i,0,:,:]
        
        attr = grad.attribute(test_data.data.float().unsqueeze(1), target=test_data.targets)
        q = [np.percentile(at[at!=0].flatten().numpy(), perc) for at in attr[:,0,:,:]]
        maskpos = torch.ones((attr.shape))
        for i in range(attr.shape[0]):
            maskpos[i,0,attr[i,0,:,:]>q[i]]=0
            test_data.data[i,:,:] = test_data.data[i,:,:]*maskpos[i,0,:,:]

    loaders = {
        'train' : torch.utils.data.DataLoader(train_data, 
                                          batch_size=batch_size, 
                                          shuffle=True, 
                                          num_workers=1),
    
        'test'  : torch.utils.data.DataLoader(test_data, 
                                          batch_size=test_batch_size, 
                                          shuffle=True, 
                                          num_workers=1),
    }
    

    
    return loaders 

