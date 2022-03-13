import argparse
import torch


def parse_training():
    parser = argparse.ArgumentParser(description='Fine-tuning densenet for image classification')
    parser.add_argument("data_dir", type=str, help='Path to folder with the data')
    parser.add_argument("data_json_dir", type=str, help='Path data.json')
    parser.add_argument("--num_layers", type=int, default=121, help='Number of layers of the densenet')
    parser.add_argument("--batch_size", type=int, default=16, help='Batch size')
    parser.add_argument("--num_epochs", type=int,  default=16, help='Number of epochs')
    parser.add_argument("--lr", type=float, default=5e-4, help='Initial learning rate')
    return parser.parse_args()


def parse_predicting():
    parser = argparse.ArgumentParser(description='Fine-tuning densenet for image classification')
    parser.add_argument("data_dir", type=str, help='Path to folder with the data')
    parser.add_argument("data_json_dir", type=str, help='Path to data.json')
    parser.add_argument("path_to_model", type=str, help='Path to pretrained model')
    return parser.parse_args()


def load_trained_model(model: torch.nn.Module, path: str) -> torch.nn.Module:
    '''
    :param model: object of the Model class
    :param path: path to the weights
    :return: trained mddel
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model
    
