import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np


def create_datasets(train_data: np.array, train_embs: np.array, train_target: np.array, 
                    test_data: np.array, test_embs: np.array, test_target: np.array, 
                    valid_pct=0.2, 
                    seed=None):
    """Converts NumPy arrays into PyTorch datasets."""

    sz = train_data.shape[0]
    idx = np.arange(sz)
    trn_idx, val_idx = train_test_split(
        idx, test_size=valid_pct, random_state=seed)
    trn_ds = TensorDataset(
        torch.tensor(train_data[trn_idx]).float(),
        torch.tensor(train_embs[trn_idx]).long(),
        torch.tensor(train_target[trn_idx]).long())
    val_ds = TensorDataset(
        torch.tensor(train_data[val_idx]).float(),
        torch.tensor(train_embs[val_idx]).long(),
        torch.tensor(train_target[val_idx]).long())
    tst_ds = TensorDataset(
        torch.tensor(test_data).float(),
        torch.tensor(test_embs).long(),
        torch.tensor(test_target).long())
    return trn_ds, val_ds, tst_ds


def create_loaders(data: TensorDataset, bs=128, jobs=0):
    """Wraps the datasets returned by create_datasets function with data loaders."""

    trn_ds, val_ds, tst_ds = data
    trn_dl = DataLoader(trn_ds, batch_size=bs, shuffle=True, num_workers=jobs)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=jobs)
    tst_dl = DataLoader(tst_ds, batch_size=bs, shuffle=False, num_workers=jobs)

    return trn_dl, val_dl, tst_dl

class DataBunch():
    def __init__(self, train_dl, valid_dl, test_dl):
        self.train_dl,self.valid_dl,self.test_dl = train_dl,valid_dl,test_dl

    @property
    def train_ds(self): return self.train_dl.dataset

    @property
    def valid_ds(self): return self.valid_dl.dataset

    @property
    def test_ds(self): return self.test_dl.dataset