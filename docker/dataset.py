import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
from skimage import color, transform


class ImagesDataset(Dataset):
    '''
    The class of the images dataset.
    '''
    def __init__(self, df: pd.DataFrame, size: tuple):
        '''
        :param df: pandas dataframe with images
        :param size: required size of images in the shape of (height, width), using in the resizing function
        '''
        super().__init__()
        self.df = df
        self.len_ = len(self.df)
        self.size = size

    def __len__(self):
        return self.len_

    @staticmethod
    def load_sample(file):
        image = Image.open(file)
        image.load()
        return image

    @staticmethod
    def transform(img, size):
        img = np.array(img)
        if img.shape[-1] == 4:
            img = color.rgba2rgb(img)
        elif len(img.shape) == 2:
            img = color.gray2rgb(img)
        img = transform.resize(img, size)
        img = img.transpose(2, 0, 1)
        img = torch.tensor(img, dtype=torch.float32)
        return img

    def __getitem__(self, index):
        img = self.load_sample(self.df.loc[index, 'path'])
        label = self.df.loc[index, 'numlabel']
        img = self.transform(img, self.size)
        return img, label

