import random
from pathlib import Path

from torch.utils.data import Dataset
from mnist import MNIST
import numpy as np
import requests
import gzip
from typing import Union
import logging
from contflame.internals import TqdmToLogger
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Permute:
    def __init__(self, in_size:tuple, tile:tuple=(1, 1), seed=1234):
        self.perm = self.get_permutation(in_size, tile, seed)
        self.tile = tile
        self.in_size = in_size

    def permute(self, img):
        if self.tile == (1, 1):
            return self.permute1d(img.reshape(self.in_size[0] * self.in_size[1]))
        else:
            return self.permute2d(img)

    def permute1d(self, img):
        if self.tile != (1, 1):
            raise ValueError('1d permutation doesn\'t support shaped permutation. tile must be (1, 1)')

        perm = self.perm.reshape(self.in_size[0] * self.in_size[1])
        aux = img[:]

        for i in range(len(img)):
            img[perm[i]] = aux[i]
        return img

    def permute2d(self, img):
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
            root (string): Directory with containing the mnist-python directory.
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

        return x, y

    def add(self, buffer, l):
        b = list(buffer)

        for i in range(l):
            self.t = self.t + b

    def _download(self, root):
        logger.info('Downloading dataset...')
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
        logger.info('Done!')

    def _setup(self, root):
        mndata = MNIST(str(root / 'mnist-python'))
        train_imgs, train_labels = mndata.load_training()
        test_imgs, test_labels = mndata.load_testing()

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


    def __init__(self, root:Union[str, Path]='.', dset:str='train', valid:float=0.0, task:int=0, tile:tuple=(1, 1), transform=None):
        """
        Args:
            root (string): Directory with containing the mnist-python directory.
            meta (bool): True - returns the meta-training dataset, False - returns the meta-test dataset
            train (bool): True - returns the training set, False - returns the test set.
                Training and test sets are internal to the meta-training and meta-test dataset.
            tasks (int): Select the tasks to keep in the dataset. If None all the tasks are used.
        """
        root = Path(root)
        self.transform = transform
        self.p = Permute((28, 28), tile=tile, seed=1234+task)

        # download and uncompress dataset if not present

        if not (root/'mnist-perm'/f'{tile[0]}_{tile[1]}_{task}.pkl').is_file():
            (root/'mnist-perm').mkdir(exist_ok=True)

            if not (root / 'mnist-python').is_dir():
                self._download(root)
            train_data, test_data = self._setup(root)

            logger.info('Permuting dataset...')
            for i in range(len(train_data)):
                x, _ = train_data[i]
                x = self.p.permute(np.array(x).reshape((28, 28)))
                train_data[i][0] = x
            for i in range(len(test_data)):
                x, _ = test_data[i]
                x = self.p.permute(np.array(x).reshape((28, 28)))
                test_data[i][0] = x
            logger.info('Done!')

            with (root/'mnist-perm'/f'{tile[0]}_{tile[1]}_{task}.pkl').open('wb') as f:
                pickle.dump((train_data, test_data), f)
        else:
            with (root/'mnist-perm'/f'{tile[0]}_{tile[1]}_{task}.pkl').open('rb') as f:
                train_data, test_data = pickle.load(f)

        if dset == 'test':
            data = test_data
        elif dset == 'train':
            data = train_data[:len(train_data) - int(len(train_data)*valid)]
        elif dset == 'valid':
            data = train_data[-int(valid*len(train_data)):]
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

        if self.transform:
            x = self.transform(x)

        return (x, y)

    def add(self, buffer, l):
        b = list(buffer)

        for i in range(l):
            self.t = self.t + b

    def _download(self, root):
        logger.info('Downloading dataset...')
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
        logger.info('Done!') # TODO add as logging

    def _setup(self, root):
        mndata = MNIST(str(root / 'mnist-python'))
        train_imgs, train_labels = mndata.load_training()
        test_imgs, test_labels = mndata.load_testing()

        train_data = [[img, labels] for img, labels in zip(train_imgs, train_labels)]
        test_data = [[img, labels] for img, labels in zip(test_imgs, test_labels)]

        return train_data, test_data


from pathlib import Path
from torch.utils.data import Dataset
from functools import reduce
import pickle
import numpy as np
import requests
import tarfile
import os


class SplitCIFAR100(Dataset):
    """Split CIFAR-100 dataset."""

    def __init__(self, root='.', meta=False, train=False, tasks=None, transform=None):
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
        if not (root / 'cifar-100-python').is_dir():
            print('Downloading dataset...')
            url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
            r = requests.get(url)

            with (root/'cifar-100-python.tar.gz').open('wb') as f:
                f.write(r.content)

            tarfile.open(str(root / 'cifar-100-python.tar.gz'), "r:gz").extractall()
            (root / "cifar-100-python.tar.gz").unlink()
            print('Done!')

        # open and unpickle train or test set
        if train:
            with (root / 'cifar-100-python/train').open('rb') as fo:
                data = pickle.load(fo)
        else:
            with (root / 'cifar-100-python/test').open('rb') as fo:
                data = pickle.load(fo)

        # transform dictionary in list of (x, y) pairs
        data = np.array([[d, l] for d, l in zip(data[b'data'], data[b'fine_labels'])])

        split = []
        for i in range(0, 100, 2):
            split.append(list(filter(lambda x: x[1] in [i, i + 1], data)))
        split = np.array(split)

        # if meta:
        #     split = split[:-20]
        # else:
        #     split = split[-20:]

        if tasks != None and max(tasks) >= len(split):
            print('Error: task index higher then number of tasks (#tasks=' + str(len(split) - 1) + ')')
        # select the required tasks
        if tasks == None:
            tasks = range(len(split))
        self.t = reduce(lambda x, y: np.concatenate((x, y)), split[tasks])

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        (x, y) = self.t[idx]

        x = x.reshape((32, 32, 3))
        if self.transform:
            x = self.transform(x)

        return (x, y)

import torchvision.transforms as transforms
import torch
from tqdm import tqdm
from contflame.data.utils import MultiLoader

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [%(levelname)-8s] %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    transform = transforms.Compose(
        [
            lambda x: torch.FloatTensor(x),
            lambda x: x.reshape((28, 28)),
            lambda x: x.unsqueeze(0)
        ])
    trainset = PermutedMNIST(dset='train', valid=0.2, transform=transform, task=0, tile=(1, 1))
    validset = PermutedMNIST(dset='valid', valid=0.2, transform=transform, task=0, tile=(1, 1))
    # trainset = SplitMNIST(dset='train', valid=0.0, transform=transform, classes=list(range(10)))
    print(len(trainset))
    print(len(validset))

    loader = MultiLoader([trainset], batch_size=256)
    for x, y in tqdm(loader):
        pass

