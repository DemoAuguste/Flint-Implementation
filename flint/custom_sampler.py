from torch.utils.data.sampler import Sampler, BatchSampler
import torch
import numpy as np


class CustomSampler(Sampler):
    """
    Customed sampler.
    Given a sequence, return sub-sequence by batch.
    """

    def __init__(self, data_source, batch_size, shuffle=False):
        # data_source: a list of indices.
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle

        if self.shuffle:
            if not isinstance(self.data_source, torch.Tensor):
                self.data_source = torch.tensor(self.data_source)
            data = self.data_source.numpy()
            np.random.shuffle(data)
            self.data_source = torch.tensor(data)

    def __iter__(self):
        return iter(self.data_source)

    def __len__(self):
        return len(self.data_source)