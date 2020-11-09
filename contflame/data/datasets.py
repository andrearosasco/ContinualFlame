import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from mnist import MNIST
import numpy as np
import requests
import gzip
from typing import Union
import logging
import pickle

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Permute:
    def __init__(self, in_size: tuple, tile: tuple = (1, 1), seed=1234):
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
                    = img[k_rows * i:k_rows * (i + 1), k_cols * j:k_cols * (j + 1)]
        return aux

    def get_permutation(self, img: tuple, kernel: tuple, seed):
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

    def __init__(self, root: Union[str, Path] = '.', dset: str = 'train', valid: float = 0.0, classes: list = None,
                 transform=None):
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
            if not (root / 'mnist-python').is_dir():
                self._download(root)
            self._setup(root)

        if dset == 'test':
            data = self.test_data
        elif dset == 'train':
            data = list(map(lambda x: x[:len(x) - int(len(x) * valid)], self.train_data))
        elif dset == 'valid':
            data = list(map(lambda x: x[-int(valid * len(x)):], self.train_data))

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
        (root / 'mnist-python').mkdir(parents=True)

        for url, fname in zip(self.urls, self.fnames):
            r = requests.get(url)
            fn = url.split('/')[-1]

            with (root / 'mnist-python' / fn).open('wb') as f:
                f.write(r.content)
            with gzip.open(str(root / 'mnist-python' / fn), 'rb') as f:
                data = f.read()
            with (root / 'mnist-python' / fn[:-3]).open('wb') as f:
                f.write(data)
            (root / 'mnist-python' / fn).unlink()
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

    def __init__(self, root: Union[str, Path] = '.', dset: str = 'train', valid: float = 0.0, task: int = 0,
                 tile: tuple = (1, 1), transform=None):
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
        self.p = Permute((28, 28), tile=tile, seed=1234 + task)

        # download and uncompress dataset if not present

        if not (root / 'mnist-perm' / f'{tile[0]}_{tile[1]}_{task}.pkl').is_file():
            (root / 'mnist-perm').mkdir(exist_ok=True)

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

            with (root / 'mnist-perm' / f'{tile[0]}_{tile[1]}_{task}.pkl').open('wb') as f:
                pickle.dump((train_data, test_data), f)
        else:
            with (root / 'mnist-perm' / f'{tile[0]}_{tile[1]}_{task}.pkl').open('rb') as f:
                train_data, test_data = pickle.load(f)

        if dset == 'test':
            data = test_data
        elif dset == 'train':
            data = train_data[:len(train_data) - int(len(train_data) * valid)]
        elif dset == 'valid':
            data = train_data[-int(valid * len(train_data)):]
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
        (root / 'mnist-python').mkdir(parents=True)

        for url, fname in zip(self.urls, self.fnames):
            r = requests.get(url)
            fn = url.split('/')[-1]

            with (root / 'mnist-python' / fn).open('wb') as f:
                f.write(r.content)
            with gzip.open(str(root / 'mnist-python' / fn), 'rb') as f:
                data = f.read()
            with (root / 'mnist-python' / fn[:-3]).open('wb') as f:
                f.write(data)
            (root / 'mnist-python' / fn).unlink()
        logger.info('Done!')  # TODO add as logging

    def _setup(self, root):
        mndata = MNIST(str(root / 'mnist-python'))
        train_imgs, train_labels = mndata.load_training()
        test_imgs, test_labels = mndata.load_testing()

        train_data = [[img, labels] for img, labels in zip(train_imgs, train_labels)]
        test_data = [[img, labels] for img, labels in zip(test_imgs, test_labels)]

        return train_data, test_data


from functools import reduce
import tarfile


class SplitCIFAR10(Dataset):
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-batches-py.tar.gz"
    label = b'labels'

    train_batches = ['data_batch_1',
                     'data_batch_2',
                     'data_batch_3',
                     'data_batch_4',
                     'data_batch_5',
                     ]
    test_batch = 'test_batch'

    no_classes = 10
    train_data, test_data = [], []

    def __init__(self, root: Union[str, Path] = '.', dset: str = 'train', valid: float = 0.0, classes: list = None,
                 transform=None):
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
        if len(self.train_data) == 0 or len(self.test_data) == 0:
            if not (root / self.base_folder).is_dir():
                self._download(root)
            self._setup(root)

        if dset == 'test':
            data = self.test_data
        elif dset == 'train':
            data = list(map(lambda x: x[:len(x) - int(len(x) * valid)], self.train_data))
        elif dset == 'valid':
            data = list(map(lambda x: x[-int(valid * len(x)):], self.train_data))

        # select the specified tasks
        if classes is not None and max(classes) >= self.no_classes:
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

    def _download(self, root):
        logger.info('Downloading dataset...')
        r = requests.get(self.url)

        with (root / self.filename).open('wb') as f:
            f.write(r.content)

        tarfile.open(str(root / self.filename), "r:gz").extractall()
        (root / self.filename).unlink()
        logger.info('Done!')

    def _setup(self, root):
        data, target = [], []
        for f in self.train_batches:
            with (root / self.base_folder / f).open('rb') as f:
                batch = pickle.load(f, encoding='bytes')
                data.extend(batch[b'data'])
                target.extend(batch[self.label])

        with (root / self.base_folder / self.test_batch).open('rb') as f:
            test = pickle.load(f, encoding='bytes')

        for i in range(self.no_classes):
            self.train_data.append(list((filter(lambda x: x[1] == i, zip(data, target)))))
            self.test_data.append(list((filter(lambda x: x[1] == i, zip(test[b'data'], test[self.label])))))

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        (x, y) = self.t[idx]

        if self.transform:
            x = self.transform(x)

        return (x, y)

class SplitCIFAR100(Dataset):
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    label = b'fine_labels'

    train_batches = ['train']
    test_batch = 'test'

    no_classes = 100
    train_data, test_data = [], []

    def __init__(self, root: Union[str, Path] = '.', dset: str = 'train', valid: float = 0.0, classes: list = None,
                 transform=None):
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
        if len(self.train_data) == 0 or len(self.test_data) == 0:
            if not (root / self.base_folder).is_dir():
                self._download(root)
            self._setup(root)

        if dset == 'test':
            data = self.test_data
        elif dset == 'train':
            data = list(map(lambda x: x[:len(x) - int(len(x) * valid)], self.train_data))
        elif dset == 'valid':
            data = list(map(lambda x: x[-int(valid * len(x)):], self.train_data))

        # select the specified tasks
        if classes is not None and max(classes) >= self.no_classes:
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

    def _download(self, root):
        logger.info('Downloading dataset...')
        r = requests.get(self.url)

        with (root / self.filename).open('wb') as f:
            f.write(r.content)

        tarfile.open(str(root / self.filename), "r:gz").extractall()
        (root / self.filename).unlink()
        logger.info('Done!')

    def _setup(self, root):
        data, target = [], []
        for f in self.train_batches:
            with (root / self.base_folder / f).open('rb') as f:
                batch = pickle.load(f, encoding='bytes')
                data.extend(batch[b'data'])
                target.extend(batch[self.label])

        with (root / self.base_folder / self.test_batch).open('rb') as f:
            test = pickle.load(f, encoding='bytes')

        for i in range(self.no_classes):
            self.train_data.append(list((filter(lambda x: x[1] == i, zip(data, target)))))
            self.test_data.append(list((filter(lambda x: x[1] == i, zip(test[b'data'], test[self.label])))))

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        (x, y) = self.t[idx]

        if self.transform:
            x = self.transform(x)

        return (x, y)


import torchvision.transforms as transforms
import torch
from contflame.data.utils import MultiLoader, Buffer

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [%(levelname)-8s] %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    train_transform = transforms.Compose(
        [
            lambda x: Image.fromarray(x.reshape((3, 32, 32)).transpose((1, 2, 0))),
            transforms.ToTensor(),
            transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0, np.array([63.0, 62.1, 66.7]) / 255.0)
        ])

    test_transform = transforms.Compose(
        [
            lambda x: Image.fromarray(x.reshape((3, 32, 32)).transpose((1, 2, 0))),
            transforms.ToTensor(),
            transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0, np.array([63.0, 62.1, 66.7]) / 255.0)
        ])

    task = list(range(1))
    trainsets = []
    memories = []

    for t in task:
        trainsets.append(SplitCIFAR100(dset='train', valid=0.2, transform=train_transform, classes=[t]))

    for m in range(1, 100):
        memories.append(Buffer(SplitCIFAR100(dset='train', valid=0.2, transform=train_transform, classes=[m]), 40))

    l = MultiLoader(trainsets + memories, batch_size=128)
    for j, (x, y) in enumerate(l):
        print(f'batch {j}')
        # for i in range(100):
        #     pass
        #     print(f'{i} -> {len(list(filter(lambda x: x == i, y)))}')

        print()