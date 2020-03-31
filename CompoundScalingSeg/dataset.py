import os
import pandas as pd
import numpy as np

from PIL import Image
from cv2 import cv2

import torch
import torchvision

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage
from torchvision import transforms, utils

from CompoundScalingSeg.model import unet_params


class NerveSegmentationDataset(Dataset):
    def __init__(self, root='./data/', no_mask_filter=True, train=True, transform=None, model_param='unet=b0'):
        super().__init__()
        self.root = root
        self.train = train
        self.transform = transform
        self.model_param = model_param

        self.items = []

        if no_mask_filter == True:
            if train == True:
                trc = pd.read_csv(root + 'train_label.csv')
                trc_ed = trc[trc['label'] == 1]
                self.items = trc_ed['image'].reset_index()
            elif train == False:
                trc = pd.read_csv(root + 'test_label.csv')
                trc_ed = trc[trc['label'] == 1]
                self.items = trc_ed['image'].reset_index()
        elif no_mask_filter == False:
            if train == True:
                trc = pd.read_csv(root + 'train_label.csv')
                self.items = trc['image'].reset_index()
            elif train == False:
                trc = pd.read_csv(root + 'test_label.csv')
                self.items = trc['image'].reset_index()

    def __getitem__(self, index):
        if self.train == True:
            src = cv2.imread('./data/train/image/' + str(self.items['image'][index]) + '.tif')
            mask = cv2.imread('./data/train/mask/' + str(self.items['image'][index]) + '_mask.tif', 0)
        elif self.train == False:
            src = cv2.imread(os.path.join('./data/test/image', str(self.items['image'][index]) + '.tif'))
            mask = cv2.imread('./data/test/mask/' + str(self.items['image'][index]) + '_mask.tif', 0)

        sample = (src, mask, unet_params(self.model_param)[-2])
        
        return sample if self.transform is None else self.transform(*sample)

    def __len__(self):
        return len(self.items)
        

class RealTestNerveSegmentationDataset(Dataset):
    def __init__(self, root='./data/', transform=None):
        super().__init__()
        self.root = root
        self.transform = transform

        self.items = []

        for i in range(1, 5509):
            self.items.append(i)

    def __getitem__(self, index):
        src = cv2.imread('./data/real_test/' + str(self.items[index]) + '.tif')
        sample = (src, index)
        
        return sample if self.transform is None else self.transform(*sample)

    def __len__(self):
        return len(self.items)
        