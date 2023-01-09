import logging
import os

import hydra
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from src.data.load_dataset import mnist
from src.models.model import MyAwesomeModel
from torch import nn
from tqdm import tqdm
from hydra.utils import get_original_cwd
import wandb as wa

log = logging.getLogger(__name__)


"""
call with python src/models/train_model.py hydra.job.chdir=True hydra.mode=MULTIRUN

from MLops_exercises/s2_organisation_and_version_control/cookie_test
"""

@hydra.main(version_base=None, config_path="../../conf", config_name="config.yaml")
def train(cfg):
    """main training function for the model, calls the subsequent training function"""
    log.info(f"configuration: \n{OmegaConf.to_yaml(cfg)}")
    log.info("Working directory : {}".format(os.getcwd()))
    hparams_ex = cfg.experiment.hparams
    hparams_model = cfg.model.hparams

    # Model Hyperparameters
    dataset_path = hparams_model.dataset_path
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = hparams_ex.batch_size
    x_dim  = hparams_model.x_dim
    hidden_dim = hparams_model.hidden_dim
    hidden_dim2 = hparams_model.hidden_dim2
    latent_dim = hparams_model.latent_dim
    lr = hparams_ex.lr
    epochs = hparams_ex.epochs
    torch.manual_seed(hparams_ex.seed)

    config = {"lr": lr, "batch_size": batch_size, "epochs": epochs, "seed": hparams_ex.seed, "hidden_dim": hidden_dim, "hidden_dim2": hidden_dim2, "latent_dim": latent_dim}
    dropout_rate = hparams_model.dropout_rate
    wa.init(project='MNIST_Cookie_test',config=config)

    model = MyAwesomeModel(x_dim, hidden_dim, hidden_dim2, latent_dim,dropout_rate).to(DEVICE)
    train_set, _ = mnist(data_path=get_original_cwd()+dataset_path,batch_size=batch_size)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model_out, loss = training(model, train_set, criterion, optimizer, epochs=epochs)
    out_str = f"{os.getcwd()}/checkpoint.pth"
    torch.save(model_out.state_dict(), out_str)
    log.info("saved to "+out_str)

    plt.figure(figsize=(10, 5))
    plt.plot(loss)
    plt.title("Training loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    wa.log({"chart": plt})
    plt.savefig("loss.png", dpi=200)


def training(model, train_set, criterion, optimizer, epochs=5,log=True):
    """Training function for the model

    Args:
        model (nn.Module): model to train
        train_set (torch.utils.data.DataLoader): training set
        criterion (nn.Module): loss function
        optimizer (torch.optim): optimizer to use for training
    Returns:
        model (nn.Module): trained model
        running_loss_l (list): list of training losses
    """
    model.train()
    pbar = tqdm(range(epochs))
    running_loss_l = []
    for e in pbar:
        running_loss = 0
        for images, labels in train_set:

            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        pbar.set_postfix({"Training loss": running_loss / len(train_set)})
        e_loss = running_loss / len(train_set)
        running_loss_l.append(e_loss)
        if log:
            wa.log({"Training loss": e_loss})
    return model, running_loss_l


if __name__ == "__main__":
    train()
