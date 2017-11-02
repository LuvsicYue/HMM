#! /nfs/raid13/babar/software/anaconda/bin/python

# -*- coding: utf-8 -*-

import os
os.chdir('/home/yunxuanli/poem_generator/data')

import numpy as np
import matplotlib.pyplot as plt
import csv
import re

with open('shakespeare.txt','r') as f_shake:
    iter_shake = csv.reader(f_shake, delimiter='\n')
    data_shake = [i for i in iter_shake]
shake = np.asarray(data_shake)

index = []
for i in range(shake.shape[0]):
    if(len(shake[i]) != 0):
        if(re.match(r'\s*\d',shake[i][0]) != None):
            index.append(i)
