# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 17:41:11 2022

@author: unknown
"""

import torch


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

