import torch
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


def evaluate(model, test_dataframe, dataset, device):
    test_dataframe['predicted'] = None
    test_dataframe['probability'] = None

    test_dataset = dataset(test_dataframe, (360, 360))

    for i in tqdm(range(len(test_dataframe))):
        img = Image.open(test_dataframe.loc[i, 'path'])
        img.load()
        img = test_dataset.transform(img).unsqueeze(0).to(device)
        res = model(img)
        y_hat = res.softmax(dim=1).argmax(dim=1)
        proba = torch.max(res.softmax(dim=1))
        test_dataframe.loc[i, 'predicted'] = y_hat.item()
        test_dataframe.loc[i, 'probability'] = proba.item()
    return test_dataframe


def get_classification_report(test_dataframe):
    y_true = list(test_dataframe['numlabel'])
    y_preds = list(test_dataframe['predicted'])
    report = classification_report(y_true, y_preds, labels=list(set(y_true)), output_dict=True)
    for clss in report.keys():
        print(clss)
        print(report[clss])
        print()


def viz_mistakes(test_dataframe):
    rslt_df = test_dataframe[test_dataframe['numlabel'] != test_dataframe['predicted']]
    print('Wrong predictions were made for the following examples:')
    print(rslt_df)
    for i in rslt_df.index:
        img = Image.open(rslt_df.loc[i, 'path'])
        img.load()
        img = np.array(img)
        plt.imshow(img)
        plt.title(f"True: {rslt_df.loc[i, 'numlabel']}, predicted: {rslt_df.loc[i, 'predicted']}")
        plt.show()
