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
Implementation of Dataset Class for Height Estimation TRAIN images

    Input parameters:
        ds_dir: csv file which includes dataset samples.
        ds_name: Dataset name (csv file name).
        classify: Redundant, keep it False.
        
    Output:
        - Dictionary includes: input image, input mask, input joint locations
        and weight input.
"""

class Images(Dataset):

    def __init__(self, ds_dir, ds_name, classify):
        self.df = pd.read_csv(ds_dir + ds_name, header=None)
        self.to_tensor = transforms.ToTensor()
        self.classify = classify

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        image_name = self.df.iloc[idx, 0]
        mask_name = self.df.iloc[idx, 1]
        joint_name = self.df.iloc[idx, 2]
        
        if self.classify:
            
            height = int(self.df.iloc[idx,3])
            
            if height < 140:
                height = 0
            elif height >= 200:
                height = 13
            else:
                height = int((height-140)/5) + 1
            
        else:
            height = torch.from_numpy(np.array([self.df.iloc[idx,3]/100])).type(torch.FloatTensor)
        
        # Reading Image 
        X = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB).astype('float32') 
        X /= 255
        X = self.to_tensor(X)
        
        # Reading Mask 
        y_mask = (cv2.imread(mask_name, 0) > 200).astype('float32')
        y_mask = to_categorical(y_mask, 2)
        y_mask = self.to_tensor(y_mask)
        
        # Reading Joint 
        y_heatmap = np.load(joint_name).astype('int64')  # For Heatmaps
        y_heatmap = torch.from_numpy(y_heatmap)

        # Reading Height
        y_height = height
        
        return {'img': X, 'mask': y_mask, 'joint': y_heatmap, 'height': y_height}