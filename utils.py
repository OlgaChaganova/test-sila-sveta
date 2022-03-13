import argparse


def parse():
    parser = argparse.ArgumentParser(description='Fine-tuning densenet for image classification')
    parser.add_argument("data_dir", type=str, help='Path to folder with the data')
    parser.add_argument("data_json_dir", type=str, help='Path data.json')
    parser.add_argument("--num_layers", type=int, default=121, help='Number of layers of the densenet')
    parser.add_argument("--batch_size", type=int, default=16, help='Batch size')
    parser.add_argument("--num_epochs", type=int,  default=16, help='Number of epochs')
    parser.add_argument("--lr", type=float, default=5e-4, help='Initial learning rate')
    return parser.parse_args()
