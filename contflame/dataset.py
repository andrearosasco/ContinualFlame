from pathlib import Path
from torch.utils.data import Dataset
from mnist import MNIST
from functools import reduce
import pickle
import numpy as np
import requests
import gzip
import os
from typing import Union


class Permute:
    def __init__(self, in_size:tuple, tile:tuple=(1, 1), seed=1234):
        self.perm = self.get_permutation(in_size, tile, seed)
        self.tile = tile
        self.in_size = in_size

    def permute(self, img):
        k_rows, k_cols = self.tile
        i_rows, i_cols = self.in_size
        if i_rows != img.shape[0] or i_cols != img.shape[1]:
            raise ValueError(f'Input dimension is {img.shape}: expected {i_rows, i_cols}')

        aux = np.zeros((i_rows, i_cols))
        t_rows, t_cols = int(i_rows / k_rows), int(i_cols / k_cols)

        perm = self.perm
        for i in range(t_rows):
            for j in range(t_cols):
                aux[k_rows * int(perm[i, j] / t_cols):k_rows * (int(perm[i, j] / t_cols) + 1),
                    k_cols * (perm[i, j] % t_cols):k_cols * (perm[i, j] % t_cols + 1)] \
                    = img[k_rows*i:k_rows*(i+1), k_cols*j:k_cols*(j+1)]
        return aux

    def get_permutation(self, img:tuple, kernel:tuple, seed):
        np.random.seed(seed)

        i_rows, i_cols = img
        k_rows, k_cols = kernel

        if i_rows % k_rows != i_cols % k_cols != 0:
            raise ValueError('One of the dimensions of the kernel do\'t divide the corresponding image dimension')

        t_rows, t_cols = int(i_rows / k_rows), int(i_cols / k_cols)
        perm = np.random.permutation(t_rows * t_cols).reshape((t_rows, t_cols))

        return perm


class SplitMNIST(Dataset):
    """Split MNIST"""

    urls = ['http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']
    fnames = ['train-data', 'train-labels', 'test-data', 'test-labels']

    no_classes = 10
    train_data, test_data = [], []

    def __init__(self, root:Union[str, Path]='.', dset:str='train', valid:float=0.0, classes:list=None, transform=None):
        """
        Args:
            root (string): Directory with containing the cifar-100-python directory.
            meta (bool): True - returns the meta-training dataset, False - returns the meta-test dataset
            train (bool): True - returns the training set, False - returns the test set.
                Training and test sets are internal to the meta-training and meta-test dataset.
            tasks (int): Select the tasks to keep in the dataset. If None all the tasks are used.
        """
        root = Path(root)
        self.transform = transform

        # download and uncompress dataset if not present
        if len(self.train_data) == len(self.test_data) == 0:
            if not (root/'mnist-python').is_dir():
                self._download(root)
            self._setup(root)

        if dset == 'test':
            data = self.test_data
        elif dset == 'train':
            data = list(map(lambda x: x[:len(x) - int(len(x)*valid)], self.train_data))
        elif dset == 'valid':
            data = list(map(lambda x: x[-int(valid*len(x)):], self.train_data))

        # if meta is not None:
        #     if meta:
        #         split = split[:-2]
        #     else:
        #         split = split[-2:]

        # select the specified tasks
        if classes != None and max(classes) >= self.no_classes:
            print('Error: Class index higher then number of classes (#classes=' + str(len(data) - 1) + ')')
        # select all the tasks (joint training)

        if classes == None:
            classes = range(len(data))

        t = []
        for i in range(len(data)):
            if i in classes:
                t += data[i]

        self.t = t
        self.l = len(self.t)

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        (x, y) = self.t[idx]
        if self.transform:
            x = self.transform(x)

        return (x, y)

    def add(self, buffer, l):
        b = list(buffer)

        for i in range(l):
            self.t = self.t + b

    def _download(self, root):
        print('Downloading dataset...')
        (root/'mnist-python').mkdir(parents=True)

        for url, fname in zip(self.urls, self.fnames):
            r = requests.get(url)
            fn = url.split('/')[-1]

            with (root/'mnist-python'/fn).open('wb') as f:
                f.write(r.content)
            with gzip.open(str(root/'mnist-python'/fn), 'rb') as f:
                data = f.read()
            with (root/'mnist-python'/fn[:-3]).open('wb') as f:
                f.write(data)
            (root/'mnist-python'/fn).unlink()
            print('Done!')

    def _setup(self, root):
        mndata = MNIST(str(root / 'mnist-python'))
        train_imgs, train_labels = mndata.load_training()
        test_imgs, test_labels = mndata.load_testing()

        train_data = []
        for i in range(self.no_classes):
            self.train_data.append(list((filter(lambda x: x[1] == i, zip(train_imgs, train_labels)))))
            self.test_data.append(list((filter(lambda x: x[1] == i, zip(test_imgs, test_labels)))))


class PermutedMNIST(Dataset):
    """Split MNIST"""

    urls = ['http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']
    fnames = ['train-data', 'train-labels', 'test-data', 'test-labels']

    no_classes = 10
    train_data, test_data = [], []

    def __init__(self, root:Union[str, Path]='.', dset:str='train', valid:float=0.0, task:int=0, tile:tuple=(1, 1), transform=None):
        """
        Args:
            root (string): Directory with containing the cifar-100-python directory.
            meta (bool): True - returns the meta-training dataset, False - returns the meta-test dataset
            train (bool): True - returns the training set, False - returns the test set.
                Training and test sets are internal to the meta-training and meta-test dataset.
            tasks (int): Select the tasks to keep in the dataset. If None all the tasks are used.
        """
        root = Path(root)
        self.transform = transform
        self.p = Permute((28, 28), tile=tile, seed=1234+task)

        # download and uncompress dataset if not present
        if len(self.train_data) == len(self.test_data) == 0:
            if not (root/'mnist-python').is_dir():
                self._download(root)
            self._setup(root)

        if dset == 'test':
            data = self.test_data
        elif dset == 'train':
            data = list(map(lambda x: x[:len(x) - int(len(x)*valid)], self.train_data))
        elif dset == 'valid':
            data = list(map(lambda x: x[-int(valid*len(x)):], self.train_data))
        else:
            raise ValueError(f'Argument type must have one of the following values: (train, test, valid)')

        # if meta is not None:
        #     if meta:
        #         split = split[:-2]
        #     else:
        #         split = split[-2:]

        # select the specified tasks

        self.t = data
        self.l = len(self.t)

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        (x, y) = self.t[idx]
        x = self.p.permute(np.array(x).reshape(28, 28))
        if self.transform:
            x = self.transform(x)

        return (x, y)

    def add(self, buffer, l):
        b = list(buffer)

        for i in range(l):
            self.t = self.t + b

    def _download(self, root):
        print('Downloading dataset...')
        (root/'mnist-python').mkdir(parents=True)

        for url, fname in zip(self.urls, self.fnames):
            r = requests.get(url)
            fn = url.split('/')[-1]

            with (root/'mnist-python'/fn).open('wb') as f:
                f.write(r.content)
            with gzip.open(str(root/'mnist-python'/fn), 'rb') as f:
                data = f.read()
            with (root/'mnist-python'/fn[:-3]).open('wb') as f:
                f.write(data)
            (root/'mnist-python'/fn).unlink()
            print('Done!')

    def _setup(self, root):
        mndata = MNIST(str(root / 'mnist-python'))
        train_imgs, train_labels = mndata.load_training()
        test_imgs, test_labels = mndata.load_testing()

        for i in range(self.no_classes):
            self.train_data = [[img, labels] for img, labels in zip(train_imgs, train_labels)]
            self.test_data = [[img, labels] for img, labels in zip(test_imgs, test_labels)]


