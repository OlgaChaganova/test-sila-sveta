import torch
import json
from sklearn import preprocessing
from data_preparing import get_dataframe
from utils import parse_predicting, load_trained_model
from model import get_model
from evaluate import evaluate, get_classification_report, viz_mistakes
from data_preparing import ImagesDataset


if __name__ == '__main__':
    args = parse_predicting()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 0. data loading
    f = open(args.data_json_dir)
    labels = json.load(f)
    print('Data loading is done.')

    # 1. data preparing
    test_data = get_dataframe(labels['test_bundle'])
    encoder = preprocessing.LabelEncoder()
    test_data['numlabel'] = encoder.fit_transform(test_data['label'])
    print('Data preparing is done.')

    # 2. model loading
    model = get_model(num_layers=121, layers_to_unfreeze=4, num_classes=3)
    model = load_trained_model(model, args.path_to_model)

    # 3. predicting
    test_data = evaluate(model=model, test_dataframe=test_data, dataset=ImagesDataset, device=device)  # saving predictions
    print('Predictions:')
    print(test_data)
    get_classification_report(test_dataframe=test_data)  # calculating metrics
    viz_mistakes(test_dataframe=test_data)
