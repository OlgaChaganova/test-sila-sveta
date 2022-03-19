import torch
import torch.nn as nn
from torchvision import models
from sklearn import preprocessing
import numpy as np
import pandas as pd

# from utils import load_sample, transform
from dataset import ImagesDataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SIZE = (360, 360)

class DenseNet:
    def __init__(self):
        print("Loading model...")
        self.model = models.densenet121(pretrained=True)
        self.model.classifier = nn.Linear(1024, 3)
        checkpoint = torch.load('trained-densenet.tr', map_location=DEVICE)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.classes_ = np.load('classes.npy', allow_pickle=True)
        print("Model is ready!")

    def predict(self, image_path):
        img = ImagesDataset.load_sample(image_path)
        img = ImagesDataset.transform(img, SIZE).unsqueeze(0)
        img = img.to(DEVICE)

        res = self.model(img)
        y_hat = res.softmax(dim=1).argmax(dim=1)
        proba = torch.max(res.softmax(dim=1))
        label = self.label_encoder.inverse_transform([y_hat])[0].split('_')
        return label[0], proba.item()
