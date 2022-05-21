# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 08:56:37 2022

@author: unknown
"""

import argparse
import json
from loaders.mnist_loader import MNIST_data
from torch.optim import Adam
from models.mnist_cnn import CNN3b
import torch
from train import train_base
import os
import numpy as np
from metrics.metrics import accuracy
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
                      
parser = argparse.ArgumentParser()
parser.add_argument("config_path", type=str, help="path to config file")
args = parser.parse_args()

with open(args.config_path, "r") as jsonfile:
    setup = json.load(jsonfile)
    
if __name__ == '__main__':
    print(setup)
    pert_acc = []
    loaders = MNIST_data(batch_size = 60)
    tmodel = torch.load(setup["model"])
    tmodel.eval()
    test_acc = accuracy(tmodel,loaders)
    pert_acc.append(test_acc)
    print("Perturbation = ", 0, "Test accuracy = ", test_acc)
    if setup["type"] == "block":
        for s in [6, 8, 10, 12]:    
            loaders = MNIST_data(square_mask=True, size_max = s)
            test_acc = accuracy(tmodel,loaders)
            pert_acc.append(test_acc)
            print("Perturbation = ", s, "Test accuracy = ", test_acc)
    
        np.save(os.path.join(setup["outfolder"], setup["type"]+"_"+setup["penalty"]+".npy"), np.array(pert_acc))
   
    elif setup["type"] == "gauss":
        for s in [0.1, 0.5, 1.0, 1.5]:    
            loaders = MNIST_data(gauss_noise = True, std = s)
            test_acc = accuracy(tmodel,loaders)
            pert_acc.append(test_acc)
            print("Perturbation = ", s, "Test accuracy = ", test_acc)
    
        np.save(os.path.join(setup["outfolder"], setup["type"]+"_"+setup["penalty"]+".npy"), np.array(pert_acc))
    else:
        print(sys.exit("Specify perturbartion!"))
