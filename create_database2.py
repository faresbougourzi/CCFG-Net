# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 18:35:57 2021

@author: Fares
"""

import torch
import numpy as np


train_set, train_fine, train_coarse  =  torch.load('./Training_Ch.pt')

training_Si = []
img1 = []
img2 = []
label1 = []
label2 = []
lab_cont = []
Coarse_label1 = []
Coarse_label2 = []

for i in range(len(train_set)):
    for j in range(i+1,len(train_set)):
        if train_coarse[i] == train_coarse[j]:
            img1.append(train_set[i])
            label1.append(train_fine[i])
            img2.append(train_set[j])
            label2.append(train_fine[j])
            Coarse_label1.append(train_coarse[i])
            Coarse_label2.append(train_coarse[j])
            if train_fine[i] == train_fine[j]:
                lab_cont.append(0)
                
            else:
                lab_cont.append(1)
        

X1 = [i for i in img1]
X2 = [i for i in img2] 
y1 = [i for i in label1] 
y2 = [i for i in label2] 
z = [i for i in lab_cont]
z1 = [i for i in Coarse_label1] 
z2 = [i for i in Coarse_label2] 
training= (X1, X2, y1, y2, z, z1, z2)
torch.save(training,'Training_Siamese_Ch.pt') 
    
