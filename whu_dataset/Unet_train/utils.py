import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import utils

from datasets.CD_dataset import CDDataset
from datasets.WHU_dataset import CDDataset_WHU


def make_numpy_grid(tensor_data, pad_value=0,padding=0):
    tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data, pad_value=pad_value,padding=padding)
    vis = np.array(vis.cpu()).transpose((1,2,0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis
