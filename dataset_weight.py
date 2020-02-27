import numpy as np 
import cv2
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from keras.utils import to_categorical
from torch.autograd import Variable

"""
Implementation of Dataset Class for Weight Estimation images

    Input parameters:
        ds_dir: csv file which includes dataset samples.
        ds_name: Dataset name (csv file name).
        normalize: Scale factor.
        classify: Redundant, keep it False.
        
    Output:
        - Dictionary includes: input image, input mask, input joint locations
        and weight input.
"""

class Images(Dataset):

    def __init__(self, ds_dir, ds_name, normalize=100, classify=False):
        self.df = pd.read_csv(ds_dir + ds_name, header=None)
        self.to_tensor = transforms.ToTensor()
        self.norm = normalize
        self.mean = np.mean(pd.read_csv('../data/REDDIT_WEIGHT_DATASET/TRAINING.csv', header=None).iloc[:,1])
        self.std = np.std(pd.read_csv('../data/REDDIT_WEIGHT_DATASET/TRAINING.csv', header=None).iloc[:,1])
        self.classify = classify
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        image_name = '../data/REDDIT_WEIGHT_DATASET/' + self.df.iloc[idx, 0]
        mask_name = '../data/Reddit-Masks/' + self.df.iloc[idx, 2]
        joint_name = '../data/Reddit-Joints/' + self.df.iloc[idx, 3]
        
        if self.classify:
            
            weight = int(np.round(self.df.iloc[idx,1]))
            
            if weight < 41:
                weight = 0
            elif weight >= 167:
                weight = 15
            else:
                weight = int((weight-41)/9) + 1
  
        else:  
            weight = torch.from_numpy(np.array([self.df.iloc[idx,1]/self.norm])).type(torch.FloatTensor)
            #weight = (torch.from_numpy(np.array([self.df.iloc[idx,1]])).type(torch.FloatTensor)-self.mean)/self.std
        
        # Reading Image 
        X = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB).astype('float32') 
        X /= 255
        X = self.to_tensor(X)
        
        # Reading Mask 
        #y_mask = np.zeros((128,128,2))
        y_mask = (cv2.imread(mask_name, 0) > 200).astype('float32')
        y_mask = to_categorical(y_mask, 2)
        y_mask = self.to_tensor(y_mask)
        
        # Reading Joint 
        y_heatmap = np.load(joint_name).astype('int64') #np.zeros((128,128))  # For Heatmaps
        y_heatmap = torch.from_numpy(y_heatmap)
        
        return {'img': X, 'mask': y_mask, 'joint': y_heatmap, 'weight': weight}