# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 16:40:16 2022

@author: unknown
"""

import torch 
from torchvision import datasets
from torchvision.transforms import ToTensor


def MNIST_data(batch_size = 20):

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


    loaders = {
        'train' : torch.utils.data.DataLoader(train_data, 
                                          batch_size=batch_size, 
                                          shuffle=True, 
                                          num_workers=1),
    
        'test'  : torch.utils.data.DataLoader(test_data, 
                                          batch_size=1, 
                                          shuffle=True, 
                                          num_workers=1),
    }
    
    return loaders 

