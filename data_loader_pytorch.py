import torch
from torch.utils.data import Dataset, DataLoader
import os
from utils import emphasis
import numpy as np

class timit_dataset(Dataset):

    def __init__(self, data_path):
        '''
        Initialization
        '''

        self.data_path = data_path
        self.filenames = [os.path.join(data_path, filename) for filename in os.listdir(data_path)]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        pair = np.load(self.filenames[idx])
        pair = emphasis(pair[np.newaxis, :, :], emph_coeff=0.95).reshape(2, -1)
        clean = pair[0].reshape(1, -1).astype('float32')
        noisy = pair[1].reshape(1, -1).astype('float32')

        return noisy, clean

def prepare_timit(data_path, batch_size=64, shuffle=True, extras={}):

    dataset = timit_dataset(data_path)
    all_indices = list(range(len(dataset)))

    if shuffle:
        np.random.shuffle(all_indices)

    sampler = torch.utils.data.sampler.RandomSampler(all_indices)

    num_workers = 0
    pin_memory = False
    # If CUDA is available
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]

    data_loader = DataLoader(dataset, batch_size = batch_size,
                             sampler = sampler,
                             num_workers = num_workers,
                             pin_memory = pin_memory)

    return data_loader
