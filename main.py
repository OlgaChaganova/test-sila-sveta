import json
from data_preparing import get_dataframe, get_dataloader, ImagesDataset
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from utils import parse_training
from model import get_model
from train import train
from evaluate import evaluate, get_classification_report, viz_mistakes
from data_preparing import ImagesDataset


if __name__ == '__main__':
    args = parse_training()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 0. data loading
    f = open(args.data_json_dir)
    labels = json.load(f)
    print('Data loading is done.')

    # 1. data preparing
    train_data = get_dataframe(labels['initial_bundle'])
    test_data = get_dataframe(labels['test_bundle'])
    valid_data, _ = train_test_split(test_data, test_size=0.5, random_state=8)

    encoder = preprocessing.LabelEncoder()
    train_data['numlabel'] = encoder.fit_transform(train_data['label'])
    valid_data['numlabel'] = encoder.fit_transform(valid_data['label'])
    test_data['numlabel'] = encoder.fit_transform(test_data['label'])

    train_loader = get_dataloader(ImagesDataset, train_data, batch_size=args.batch_size, size=(360, 360), shuffle=True)
    valid_loader = get_dataloader(ImagesDataset, valid_data.reset_index(), batch_size=args.batch_size, size=(360, 360), shuffle=True)
    print('Data preparing is done.')

    # 2. model selection
    model = get_model(num_layers=args.num_layers, layers_to_unfreeze=4, num_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=1, eta_min=args.lr/100)
    print('Model selection is done.')

    # 3. training
    print('Training has started.')
    history, model = train(model=model,
                           opt=optimizer,
                           scheduler=lr_scheduler,
                           criterion=nn.CrossEntropyLoss(),
                           train_loader=train_loader,
                           valid_loader=valid_loader,
                           epochs=args.num_epochs,
                           device=device,
                           start_idx=0)

    print('Training is done.')

    # 4. evaluating
    test_data = evaluate(model=model, test_dataframe=test_data, dataset=ImagesDataset, device=device)  # saving predictions
    print('Predictions:')
    print(test_data)
    get_classification_report(test_dataframe=test_data)  # calculating metrics
    viz_mistakes(test_dataframe=test_data)
