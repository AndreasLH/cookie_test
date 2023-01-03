import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def mnist(data_path: str = "data/processed/", eval: bool = False):
    """Pytorch version of the mnist dataset

        returns a train and test loader"""
    # exchange with the corrupted mnist dataset
    if not eval:
        train = np.load(data_path + "train.npz")
        images = train["images"]
        labels = train["labels"]

        test = np.load(data_path + "test.npz")
    else:
        test = np.load(data_path)

    class Train_dataset(Dataset):
        def __init__(self):
            self.data = torch.from_numpy(images).view(-1, 1, 28, 28)
            self.label = torch.from_numpy(labels)

        def __getitem__(self, index):
            return self.data[index].float(), self.label[index]

        def __len__(self):
            return len(self.data)

    if not eval:
        train_dataset = Train_dataset()
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    class Test_dataset(Dataset):
        def __init__(self):
            self.data = torch.tensor(test["images"]).view(-1, 1, 28, 28)
            self.label = torch.tensor(test["labels"])

        def __getitem__(self, index):
            return self.data[index].float(), self.label[index]

        def __len__(self):
            return len(self.data)

    test_dataset = Test_dataset()
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    if not eval:
        return train_loader, test_loader
    else:
        return None, test_loader
