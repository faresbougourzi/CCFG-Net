#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 11:56:35 2021

@author: walter
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 00:22:40 2021

@author: Fares
"""
from torchvision.datasets import MNIST
#import warnings
import PIL 
import os
import os.path
import numpy as np
import torch
import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms
#########################################

class Letter_loader(MNIST):

    def __init__(self, root, train, transform=None, target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set
        self.data, self.y1, self.y2 = torch.load(os.path.join(self.root, self.train))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, y1, y2 = self.data[index], self.y1[index], self.y2[index]        

        img1 = np.array(img)
        # img1 = img.transpose((1, 2, 0))       
        
        img1 = img1.astype(np.uint8)        
              
        if self.transform is not None:
            img1 = self.transform(img1)
            
        if self.target_transform is not None:
            y1 = self.target_transform(y1)
            
        return   img1, y1, y2

    def __len__(self):
        return len(self.data)
    
#########################################
    
class Letter_loader2(MNIST):

    def __init__(self, root, train, transform=None, target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set
        self.data, self.y1, self.y2 = torch.load(os.path.join(self.root, self.train))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, y1, y2 = self.data[index], self.y1[index], self.y2[index]        

        img1 = np.array(img)
        # img1 = img.transpose((1, 2, 0))       
        
        img1 = img1.astype(np.uint8)        
              
        if self.transform is not None:
            img1 = self.transform(img1)
            
        if self.target_transform is not None:
            y1 = self.target_transform(y1)
            
        return   img1, y1, y2

    def __len__(self):
        return len(self.data)     
#########################################

class Letter_loaderC(MNIST):

    def __init__(self, root, train, transform=None, target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set
        self.data, self.y1, self.y2 = torch.load(os.path.join(self.root, self.train))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img1, y1, y2 = self.data[index], self.y1[index], self.y2[index] 
        
        # img1 = np.squeeze(img1)# 
        # print(img1.shape)
     
              
        if self.transform is not None:
            img1 = self.transform(img1)
            
        if self.target_transform is not None:
            y1 = self.target_transform(y1)
            
        return   img1, y1, y2

    def __len__(self):
        return len(self.data)    
    
#########################################
class Letter_loader2im(MNIST):

    def __init__(self, root, train, transform=None, target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set
        self.data1,self.data2, self.y1, self.y2, self.z= torch.load(os.path.join(self.root, self.train))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img1,img2, y1, y2,z= self.data1[index], self.data2[index], self.y1[index], self.y2[index], self.z[index]        

        img1 = np.array(img1)
        img2 = np.array(img2)
        # img1 = img.transpose((1, 2, 0))       
        
        img1 = img1.astype(np.uint8) 
        img2 = img2.astype(np.uint8)  
              
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        if self.target_transform is not None:
            y1 = self.target_transform(y1)
            y2 = self.target_transform(y2)
            
        return   img1,img2, y1, y2,z

    def __len__(self):
        return len(self.data1)
    
#########################################
from torch.utils import data
   
class Letter_loaderSia(data.Dataset):

    def __init__(self, list_IDs, path, transform=None):
        
        self.list_IDs = list_IDs      
        self.database_path = path
        self.transform = transform 
        self.lines_train = []
        with open('./Siamese_train.txt','r') as data_file:
            for line in data_file:
                self.lines_train.append(line.split(','))         

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """                
        imgn1,imgn2, label1, label2, lab_cont= self.lines_train[self.list_IDs[index]]
        y1 = int(label1)
        y2 = int(label2)
        z = int(lab_cont)
        im_path1 = os.path.join(self.database_path, label1, imgn1)
        im_path2 = os.path.join(self.database_path, label2, imgn2)
        # print(im_path2)

        img1 = cv2.imread(im_path1)
        img2 = cv2.imread(im_path2)
        
              
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
            
        return   img1, img2, y1, y2, z

    def __len__(self):
        return len(self.list_IDs)
#########################################
from torch.utils import data
   
class Letter_loaderSia2(data.Dataset):

    def __init__(self, list_IDs, path, data, transform=None):
        
        self.list_IDs = list_IDs      
        self.database_path = path
        self.transform = transform 
        self.lines_train = data
        # with open('./Siamese_train.txt','r') as data_file:
        #     for line in data_file:
        #         self.lines_train.append(line.split(','))         

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """                
        imgn1,imgn2, label1, label2, lab_cont= self.lines_train[self.list_IDs[index]]
        y1 = int(label1)
        y2 = int(label2)
        z = int(lab_cont)
        im_path1 = imgn1
        im_path2 = imgn2
        # print(im_path2)# , label1, , label2,self.database_path self.database_path

        img1 = cv2.imread(im_path1)
        img2 = cv2.imread(im_path2)
        
              
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
            
        return   img1, img2, y1, y2, z

    def __len__(self):
        return len(self.list_IDs)    
    
    
 #####################################
from torch.utils import data

class Letters_loader_pt(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, path, transform=None):
        'Initialization'
        self.list_IDs = list_IDs
        self.path = path
        self.transform = transform

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        pth = self.path

        # Load data and get label
        # X, y1, y2, y3, y4  = torch.load(pth +'\\'+ ID + '.pt') 
        X1, X2, y1, y2, y3, y4, y5  = torch.load(os.path.join(pth, ID + '.pt'))
        # print(X.shape)
        # print(ID)

        
        if self.transform is not None:
            X1 = self.transform(X1)
            X2 = self.transform(X2)

        return X1, X2, y1, y2, y3, y4, y5    
    
    
#####################################

   
#########################################


class Letter_loader2im2(MNIST):

    def __init__(self, root, train, transform=None, target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set
        self.data1,self.data2, self.y1, self.y2, self.z, self.z1, self.z2= torch.load(os.path.join(self.root, self.train))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img1,img2, y1, y2,z, z1, z2= self.data1[index], self.data2[index], self.y1[index], self.y2[index], self.z[index], self.z1[index], self.z2[index]        

        img1 = np.array(img1)
        img2 = np.array(img2)
        # img1 = img.transpose((1, 2, 0))       
        
        img1 = img1.astype(np.uint8) 
        img2 = img2.astype(np.uint8)  
              
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        if self.target_transform is not None:
            y1 = self.target_transform(y1)
            y2 = self.target_transform(y2)
            
        return   img1,img2, y1, y2,z, z1, z2

    def __len__(self):
        return len(self.data1) 
    
    
#########################################
class Letter_loader2im2new(MNIST):

    def __init__(self, root, train, transform=None, target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set
        self.data1,self.data2, self.y1, self.y2, self.z, self.z1, self.z2, self.z3, self.z4= torch.load(os.path.join(self.root, self.train))
# img1, img2, lab1, lab2, pairspone, lab_inter, lab_uni, coarselab1, coarselab2
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img1,img2, y1, y2,z, z1, z2, z3, z4= self.data1[index], self.data2[index], self.y1[index], self.y2[index], self.z[index], self.z1[index], self.z2[index] , self.z3[index], self.z4[index]       

        # img1 = np.array(img1)
        # img2 = np.array(img2)
        # # img1 = img.transpose((1, 2, 0))       
        
        # img1 = img1.astype(np.uint8) 
        # img2 = img2.astype(np.uint8)  
              
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        if self.target_transform is not None:
            y1 = self.target_transform(y1)
            y2 = self.target_transform(y2)
            
        return   img1,img2, y1, y2,z, z1, z2, z3, z4

    def __len__(self):
        return len(self.data1)        
#########################################
class Letter_loader2im3(MNIST):

    def __init__(self, root, train, transform=None, target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set
        self.data1,self.data2, self.y1, self.y2, self.z, self.z1, self.z2, self.z3, self.z4= torch.load(os.path.join(self.root, self.train))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img1,img2, y1, y2,z, z1, z2, z3, z4= self.data1[index], self.data2[index], self.y1[index], self.y2[index], self.z[index], self.z1[index], self.z2[index], self.z3[index], self.z4[index]        

        img1 = np.array(img1)
        img2 = np.array(img2)
        # img1 = img.transpose((1, 2, 0))       
        
        img1 = img1.astype(np.uint8) 
        img2 = img2.astype(np.uint8)  
              
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        if self.target_transform is not None:
            y1 = self.target_transform(y1)
            y2 = self.target_transform(y2)
            
        return   img1,img2, y1, y2,z, z1, z2, z3, z4

    def __len__(self):
        return len(self.data1)    

#####################################
