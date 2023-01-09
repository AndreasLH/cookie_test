from tests import _PATH_DATA

from src.data.load_dataset import mnist
from src.models.model import MyAwesomeModel
import torch
import numpy as np
import pytest

batch_size = 64
model = MyAwesomeModel()

@pytest.mark.parametrize("test_input, expected", [(torch.randn(64,1,28,28), (64,10)),(torch.randn(1,1,28,28), (1,10)), pytest.param(torch.randn(64,1,26,26), (64,10), marks=pytest.mark.xfail)])
def test_shape_in_out_model(test_input, expected):
    # test_input = torch.randn(64,1,28,28)
    out = model(test_input)
    assert out.shape == expected, f"Expected output to be of shape (64,10), but was {out.shape}"

def test_error_on_wrong_shape():
   with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
      model(torch.randn(1,2,3))