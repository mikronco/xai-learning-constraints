# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 17:41:11 2022

@author: unknown
"""

import numpy as np
import torch
from captum.attr import Saliency
from torch.nn import Softmax 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def accuracy(model, loaders):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loaders['test']:
            images, labels = images.to(device), labels.to(device)
            test_output = model(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            correct += (pred_y == labels).sum().item()
            pass
    acc = correct/len(loaders["test"])
    return acc



def MoRF(model, loaders, perc):
    softm = Softmax()
    drops = np.zeros(len(perc))
    grad = Saliency(model)
    for images, labels in loaders['test']:
        images, labels = images.to(device), labels.to(device)
        attr = grad.attribute(images, target=labels.item())
        attr = attr.squeeze()
        for j in range(len(perc)):
            q3, q1 = np.percentile(attr[attr!=0].flatten().cpu().detach().numpy(), [perc[j], 100-perc[j]])
            maskpos = torch.ones((images.shape[-1], images.shape[-1]))
            maskpos[attr>q3]=0
            d = softm(model(images.cuda()*maskpos.cuda())).cpu().detach().numpy()[0,torch.argmax(softm(model(images.cuda())))]/torch.max(softm(model(images.cuda()))).cpu().detach().numpy()
            drops[j] += d
    
    return drops/len( len(loaders['test']))