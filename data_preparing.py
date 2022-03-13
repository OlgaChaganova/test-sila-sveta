import torch
from torch.utils.data import Dataset, DataLoader
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

    def load_sample(self, file):
        image = Image.open(file)
        image.load()
        return image

    def transform(self, img):
        img = np.array(img)
        if img.shape[-1] == 4:
            img = color.rgba2rgb(img)
        elif len(img.shape) == 2:
            img = color.gray2rgb(img)
        img = transform.resize(img, self.size)
        img = img.transpose(2, 0, 1)
        img = torch.tensor(img, dtype=torch.float32)
        return img

    def __getitem__(self, index):
        img = self.load_sample(self.df.loc[index, 'path'])
        label = self.df.loc[index, 'numlabel']
        img = self.transform(img)
        return img, label


def get_dataframe(labels: list):
    '''
    Function for creating the dataframe with given images.
    :param labels: list of labels from the data.json
    :return: dataframe with the path to image, its label and size.
    '''
    df = pd.DataFrame(columns=['path', 'label', 'size'])
    for i in range(len(labels)):
        if labels[i]['category'] is not None:
            if labels[i]['category']['name'] == 'Plants':
                df.loc[len(df.index), ['path', 'label']] = [labels[i]['file'], 'plants']
            elif labels[i]['category']['name'] == 'Vehicles':
                df.loc[len(df.index), ['path', 'label']] = [labels[i]['file'], 'vehicle']
            else:
                df.loc[len(df.index), ['path', 'label']] = [labels[i]['file'], 'other']
        else:
            df.loc[len(df.index), ['path', 'label']] = [labels[i]['file'], 'other']
        img = Image.open(labels[i]['file'])
        img.load()
        df.loc[len(df.index)-1, 'size'] = np.array(img).shape
    return df


def get_dataloader(dataset: torch.utils.data.Dataset,
                   df: pd.DataFrame,
                   size: tuple,
                   batch_size: int,
                   shuffle: bool) -> torch.utils.data.DataLoader:
    '''
    Function for creating a dataloader with images.
    :param dataset: class of the torch dataset
    :param df: pandas dataframe with images
    :param size: required size of images in the shape of (height, width)
    :param batch_size: the size of the batch of data
    :param shuffle: boolean (True if need to shuffle the examples in the dataloader)
    :return: dataloader with data
    '''
    data_set = dataset(df, size)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    return data_loader
