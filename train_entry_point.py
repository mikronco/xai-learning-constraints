# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 08:56:37 2022

@author: unknown
"""

import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument("config_path", type=str, help="path to config file")
args = parser.parse_args()
print(args.config_path)

with open(args.config_path, "r") as jsonfile:
    setup = json.load(jsonfile)
    
print(setup)