from tests import _PATH_DATA

from src.data.load_dataset import mnist
from src.models.model import MyAwesomeModel
from src.models.train_model import training
import torch
import numpy as np

batch_size = 64
train_loader, test_loader = mnist(data_path="data/processed/", batch_size=batch_size)
model = MyAwesomeModel()

def test_shape_in_out_model():
    train_loader, _ = mnist(data_path="data/processed/",batch_size=batch_size)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    model_out, loss = training(model, train_loader, criterion, optimizer, epochs=1, log=False)
    assert isinstance(model_out, MyAwesomeModel), "returned model was not a model" # get a correct model back
    assert loss[0] > -1e6 and loss[0] < 1e6, "Loss was either to high or low (<-1e6, >1e6)" # loss is reasonable
