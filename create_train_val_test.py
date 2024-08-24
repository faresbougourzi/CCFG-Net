# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 16:56:29 2021

@author: Fares
"""

import os
from skimage import  transform 
import scipy.io as sio
import dlib
import numpy as np
import cv2
import os
import torch

import tqdm as tqdm 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


import re
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

database_path = '.\data'
train_spt = 'train'
val_spt = 'val'
test_spt = 'test'


Training_data = []
Validation_data = []
Testing_data = []

Training_label = []
Validation_label = []
Testing_label = []

Training_labelc = []
Validation_labelc = []
Testing_labelc = []



data_splits = os.listdir(database_path)
for split in data_splits:
    if split == train_spt:
        split_dir = os.path.join(database_path,split)
        coarse_files = sorted_alphanumeric(os.listdir(split_dir))
        cls = -1
        cls_co = -1
        for coarse_class in coarse_files:
            coarse_classes_path = os.path.join(split_dir, coarse_class)
            coarse_classes = sorted_alphanumeric(os.listdir(coarse_classes_path))
            cls_co += 1
            for fine_class in coarse_classes:
                images_names = sorted_alphanumeric(os.listdir(os.path.join(coarse_classes_path, fine_class)))
                cls += 1
                for image in images_names:
                    spl_path = os.path.join(split_dir,coarse_class,fine_class)
                    im_path = os.path.join(spl_path, image)
                    img = cv2.imread(im_path)
                    Training_data.append(cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA))
                    Training_label.append(cls)
                    Training_labelc.append(cls_co)
                    
    elif  split == val_spt:
        split_dir = os.path.join(database_path,split)
        coarse_files = sorted_alphanumeric(os.listdir(split_dir))
        cls = -1
        cls_co = -1
        for coarse_class in coarse_files:
            coarse_classes_path = os.path.join(split_dir, coarse_class)
            coarse_classes = sorted_alphanumeric(os.listdir(coarse_classes_path))
            cls_co += 1
            for fine_class in coarse_classes:
                images_names = sorted_alphanumeric(os.listdir(os.path.join(coarse_classes_path, fine_class)))
                cls += 1
                for image in images_names:
                    spl_path = os.path.join(split_dir,coarse_class,fine_class)
                    im_path = os.path.join(spl_path, image)
                    img = cv2.imread(im_path)
                    Validation_data.append(cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA))
                    Validation_label.append(cls)
                    Validation_labelc.append(cls_co)
                    
    elif  split == test_spt:
        split_dir = os.path.join(database_path,split)
        coarse_files = sorted_alphanumeric(os.listdir(split_dir))
        cls = -1
        cls_co = -1
        for coarse_class in coarse_files:
            coarse_classes_path = os.path.join(split_dir, coarse_class)
            coarse_classes = sorted_alphanumeric(os.listdir(coarse_classes_path))
            cls_co += 1
            for fine_class in coarse_classes:
                images_names = sorted_alphanumeric(os.listdir(os.path.join(coarse_classes_path, fine_class)))
                cls += 1
                for image in images_names:
                    spl_path = os.path.join(split_dir,coarse_class,fine_class)
                    im_path = os.path.join(spl_path, image)
                    img = cv2.imread(im_path)
                    Testing_data.append(cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA))
                    Testing_label.append(cls)
                    Testing_labelc.append(cls_co)    

X = torch.Tensor([i for i in Training_data]) 
y = torch.Tensor([i for i in Training_label]) 
z = torch.Tensor([i for i in Training_labelc])
training= (X, y, z)
torch.save(training,'Training_Ch.pt') 


X = torch.Tensor([i for i in Validation_data]) 
y = torch.Tensor([i for i in Validation_label]) 
z = torch.Tensor([i for i in Validation_labelc])
training= (X, y, z)
torch.save(training,'Validation_Ch.pt') 

                    

X = torch.Tensor([i for i in Testing_data]) 
y = torch.Tensor([i for i in Testing_label]) 
z = torch.Tensor([i for i in Testing_labelc])
training= (X, y, z)
torch.save(training,'Testing_Ch.pt')  
