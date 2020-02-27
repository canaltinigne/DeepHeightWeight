import numpy as np 
import cv2
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from torch.autograd import Variable

"""
Implementation of Dataset Class for Weight Estimation TEST images

    Input parameters:
        reddit: Use Reddit dataset, keep it True since we don't use W8-300 dataset.
        
    Output:
        - Dictionary includes: input image, weight input and image name.
"""

class TestImages(Dataset):

    def __init__(self, reddit=True):
        
        if reddit:
            self.df = pd.read_csv('../data/REDDIT_WEIGHT_DATASET/TEST_40-120.csv', header=None)
            self.tag = 'REDDIT_WEIGHT_DATASET/'
        else:
            self.df = pd.read_csv('../data/W8-300/TEST.csv', header=None)
            self.tag = 'W8-300/'
        
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        image_name = '../data/' + self.tag + self.df.iloc[idx, 0]
        height = torch.from_numpy(np.array([self.df.iloc[idx,1]])).type(torch.FloatTensor)

        # Reading Image 
        X = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB).astype('float32') 
        X /= 255
        X = self.to_tensor(X)
        
        return {'img': X, 'weight': height, 'name': self.df.iloc[idx, 0]}