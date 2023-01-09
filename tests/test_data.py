"""
dataset = MNIST(...)
assert len(dataset) == N_train for training and N_test for test
assert that each datapoint has shape [1,28,28] or [768] depending on how you choose to format
assert that all labels are represented
"""

from tests import _PATH_DATA
import os.path

from src.data.load_dataset import mnist
import numpy as np
import pytest

batch_size = 64
data_path = "data/processed/"
if os.path.exists(data_path):
    train_loader, test_loader = mnist(data_path=data_path, batch_size=batch_size)

N_train = 25000
N_test = 5000
all_labels = list(range(10))

@pytest.mark.skipif(not os.path.exists(data_path), reason="Data files not found")
def test_length_test_set():
    assert len(test_loader.dataset) == N_test, f"Test dataset was not the correct length, expected {N_test}, but was {len(test_loader.dataset)}"

@pytest.mark.skipif(not os.path.exists(data_path), reason="Data files not found")
def test_length_train_set():
    assert len(train_loader.dataset) == N_train, f"Train dataset was not the correct length, expected {N_train}, but was {len(train_loader.dataset)}"

@pytest.mark.skipif(not os.path.exists(data_path), reason="Data files not found")
def test_shape_train():
    for img, label in train_loader:
        assert img.shape[1:4] == (1,28,28), "Image shape incorrect"

@pytest.mark.skipif(not os.path.exists(data_path), reason="Data files not found")
def test_shape_test():
    for img, label in test_loader:
        assert img.shape[1:4] == (1,28,28), "Image shape incorrect"

@pytest.mark.skipif(not os.path.exists(data_path), reason="Data files not found")
def test_all_labels_represented_train():
    met_labels = []
    for _, labels in train_loader:
        for label in labels:
            if label not in met_labels:
                met_labels.append(label.item())
        if np.all(np.sort(np.array(met_labels)) == np.array(all_labels)):
            break
    assert np.all(np.sort(np.array(met_labels)) == np.array(all_labels)), "Not all labels present"

@pytest.mark.skipif(not os.path.exists(data_path), reason="Data files not found")
def test_all_labels_represented_test():
    met_labels = []
    for _, labels in test_loader:
        for label in labels:
            if label not in met_labels:
                met_labels.append(label.item())
        if np.all(np.sort(np.array(met_labels)) == np.array(all_labels)):
            break
    assert np.all(np.sort(np.array(met_labels)) == np.array(all_labels)), "Not all labels present"
