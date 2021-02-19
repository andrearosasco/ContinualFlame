import random
from typing import Union

import torch
from torch.utils.data import DataLoader


class MultiLoader:

    def __init__(self, datasets: list, batch_size: Union[int, list], pin_memory=True):
        '''
        Extension of the PyTorch dataloader. The main feature is the ability
        to create the returned minibatches by sampling from different datasets.
        The iterator stops returning elements when each of them was returned at least once.
        e.g. If dataset A has 1000 elements, dataset B has 100 and we specify batch_size=10
             each mini batch will contain 5 elements from dataset A and 5 from dataset B.
             A total of 2000 elements will be returned, the elements from dataset A will be
             returned just once, the elements from dataset B will be returned multiple times.
             In this case, as long as the batch_size <= 200 there won't be repated elements
             inside the mini batch.

        :param datasets: list of datasets used to create the minibatches.
        :param batch_size: if it's an int batches of the specified size
            are returned. The returned batches are composed by sampling
            from each dataset batch_size / len(datasets) elements.
            If batch_size is a list it can be used to specify how many
            elements to sample from each dataset.
        '''
        self.datasets = []
        self.no_datasets = len(datasets)
        self.no_steps = 0

        if type(batch_size) == int:
            b = batch_size
            q = int(batch_size / len(datasets))
            r = batch_size % len(datasets)
            batch_size = [q + 1 if x < r else q for x in range(self.no_datasets)]
            assert sum(batch_size) == b


        self.batch_size = batch_size

        for i, ds in enumerate(datasets):
            dl = DataLoader(ds, batch_size=batch_size[i], shuffle=True, pin_memory=pin_memory)
            self.no_steps = len(dl) if len(dl) > self.no_steps else self.no_steps
            self.datasets.append(dl)

    def __next__(self):
        if self.actual_steps == self.no_steps:
            raise StopIteration

        batch_in = batch_out = None

        for i in range(len(self.iters)):
            x = y = None
            while x is None or x.size(0) < self.batch_size[i]:
                try:
                    inp, out = next(self.iters[i])
                    x = torch.cat((x, inp)) if x is not None else inp
                    y = torch.cat((y, out)) if y is not None else out

                except StopIteration:
                    self.iters[i] = self.datasets[i].__iter__()

            x = x[:self.batch_size[i]]
            y = y[:self.batch_size[i]]

            batch_in = torch.cat((batch_in, x)) if batch_in != None else x
            batch_out = torch.cat((batch_out, y)) if batch_out != None else y

        self.actual_steps += 1
        return batch_in, batch_out

    def __iter__(self):
        self.iters = []
        self.actual_steps = 0

        for ds in self.datasets:
            self.iters.append(ds.__iter__())

        return self

    def __len__(self):
        return self.no_steps


class Buffer:

    def __init__(self, ds, dim, transform=None):
        self.transform = transform
        l = len(ds)
        r = []

        if dim < 0:
            for i in range(len(ds)):
                r.append(ds[i])
        else:
            for i in range(dim):
                r.append(ds[i])

            for i in range(dim, l):
                h = random.randint(0, i)
                if h < dim:
                    r[h] = ds[i]
        self.r = r

    def __getitem__(self, item):
        (x, y) = self.r[item]

        if self.transform:
            x = self.transform(x)

        return (x, y)

    def __setitem__(self, key, value):
        self.r[key] = value

    def __len__(self):
        return len(self.r)

    def __add__(self, buffer):
        self.r = self.r + list(buffer)
        return self

