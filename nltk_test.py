#! /nfs/raid13/babar/software/anaconda/bin/python
# coding: utf-8

import sys
sys.path.append('/home/yunxuanli/poem_generator/')
from data.data_import import *
import numpy as np
import nltk


"""
import re
import csv

with open('shakespeare.txt','r') as f_shake:
    iter_shake = csv.reader(f_shake, delimiter='\n')
    data_shake = [i for i in iter_shake]
shake = np.asarray(data_shake)

# find head of each poem
index = []
for i in range(shake.shape[0]):
	if(len(shake[i]) != 0):
		if(re.match(r'\s*\d',shake[i][0]) != None):
			index.append(i)
"""


A = []
for i in range(len(index)):
	A.append(map(lambda x:(x,''), nltk.word_tokenize(shake[index[i]+1][0])))
	A.append(map(lambda x:(x,''), nltk.word_tokenize(shake[index[i]+3][0])))
#print len(A)
symbols = list(set([ss[0] for sss in A for ss in sss]))

states = range(15)
trainer = nltk.tag.hmm.HiddenMarkovModelTrainer(states=states,symbols=symbols)
m = trainer.train_unsupervised(A)
print m.random_sample(np.random,10)


