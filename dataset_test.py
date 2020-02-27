import numpy as np 
import cv2
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from torch.autograd import Variable

"""
Implementation of Dataset Class for Height Estimation TEST images

    Input parameters:
        semih_model: In order to use the same test images to compare 
                     our results with another paper.
        our_norm: Redundant parameter.
        
    Output:
        - Dictionary includes: input image, input face image (if dataset 
                               from other paper is used), input height,
                               input mask and joint locations.
"""

class TestImages(Dataset):

    def __init__(self, semih_model, our_norm):
        
        if not semih_model:
            self.df = pd.read_csv('../data/IMDB_PAPER_DATASET_39K/TEST.csv', header=None)
        else:
            self.df = pd.read_csv('/data/SemesterProjectFinal_Can/data/IMDB_SEMIH_DATASET/TEST.csv', header=None)
            
        self.to_tensor = transforms.ToTensor()
        self.height_mean = 171.8
        self.height_std = 10.0
        self.our_norm = our_norm
        self.semih_model = semih_model

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        image_name = self.df.iloc[idx, 0]
        mask_name = self.df.iloc[idx, 1]
        joint_name = self.df.iloc[idx, 2]

        if self.semih_model:
            
            height = torch.from_numpy(np.array([self.df.iloc[idx,2]])).type(torch.FloatTensor)
            face_name = self.df.iloc[idx, 1]
            
            X = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB).astype('float32')
            X = ((X / 255) - 0.485) / 0.229
            X = self.to_tensor(X).type(torch.FloatTensor)
            
            y_face = cv2.imread(face_name, 0)
            y_face = ((y_face / 255) - 0.485) / 0.229 
            y_face = self.to_tensor(y_face).type(torch.FloatTensor)
            
        else:
            
            height = torch.from_numpy(np.array([self.df.iloc[idx,3]])).type(torch.FloatTensor)

            # Reading Image 
            X = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB).astype('float32') 
            X /= 255
            X = self.to_tensor(X)
            
            y_face = 1

        
        return {'img': X, 'image_face': y_face, 'height': height, 'mask': cv2.imread(mask_name, 0),
               'joint': np.load(joint_name)}