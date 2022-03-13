# test-sila-sveta
Test task.
The pretrained DenseNet with further fine-tuning  is used for classification of images into three classes: plants, vehicles and others. 

To clone this repository on your local computer, run in terminal:
```
git clone https://github.com/OlgaChaganova/test-sila-sveta.git
```
To install all required dependencies, run:

```
pip install -r requirements.txt
```

To start an experiment, run:
```
python main.py data_dir data_json_dir --num_layers --batch_size --num_epochs --lr
```
where 
- `data_dir` is the path to folder with the images;
- `data_json_dir` is the path to data.json;
- `--num_layers` is number of layers of the densenet used for classification;
- `--batch_size` is batch size;
- `--num_epochs` is number of epochs for training;
- `--lr` is initial learning rate.

Hyperparameters can be customized, but you can use the following: 
- `num_layers` = 121
- `batch_size` = 16
- `num_epochs` = 8
- `lr` = 5e-4
