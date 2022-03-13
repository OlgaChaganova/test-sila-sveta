import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm import tqdm


def train(model: torch.nn.Module,
          opt: torch.optim,
          scheduler: torch.optim.lr_scheduler,
          criterion: torch.nn,
          train_loader: torch.utils.data.DataLoader,
          valid_loader: torch.utils.data.DataLoader,
          epochs: int,
          device: torch.device,
          start_idx: int) -> (dict, torch.nn.Module):
    '''
    Training loop.
    :param model: model to be trained
    :param opt: optimizer
    :param scheduler: learning rate scheduler
    :param criterion: loss function
    :param train_loader: dataloader with the examples from the training set
    :param valid_loader: dataloader with the examples from the validation set
    :param epochs: number of training epochs
    :param device: device for training
    :param start_idx: start number of the epoch
    :return: dict with the history of the model training and trained model.
    '''
    device = torch.device(device)
    model = model.to(device)
    history = {'train loss by epoch': [],
               'valid loss by epoch': [],
               'train loss by batch': [],
               'valid loss by batch': []}

    iters = len(train_loader)

    log_template = "Epoch {ep:d}:\t mean train_loss: {t_loss:0.6f}\t mean val_loss {v_loss:0.6f}\n"

    for i, epoch in enumerate(range(epochs)):
        model.train()

        train_loss_per_epoch = []
        val_loss_per_epoch = []

        for x, y in tqdm(train_loader):
            opt.zero_grad()
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            train_loss_per_epoch.append(loss.item())

            loss.backward()
            opt.step()

            if scheduler is not None:
                scheduler.step(epoch + i / iters)

        history['train loss by epoch'].append(np.mean(train_loss_per_epoch))
        history['train loss by batch'].extend(train_loss_per_epoch)

        model.eval()
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss_per_epoch.append(loss.item())

            curr_loss_val = np.mean(val_loss_per_epoch)
            history['valid loss by epoch'].append(curr_loss_val)
            history['valid loss by batch'].extend(val_loss_per_epoch)

        clear_output(True)
        fig, axs = plt.subplots(1, 2, figsize=(20, 5))

        axs[0].plot(history['train loss by batch'])
        axs[0].set_title("Train loss")
        axs[0].set_xlabel("Batch")
        axs[0].set_ylabel("Loss")

        axs[1].plot(history['valid loss by batch'])
        axs[1].set_title("Valid loss")
        axs[1].set_xlabel("Batch")
        axs[1].set_ylabel("Loss")
        plt.show()

        tqdm.write(log_template.format(ep=start_idx + epoch + 1, t_loss=np.mean(train_loss_per_epoch),
                                       v_loss=curr_loss_val))

    return history, model
