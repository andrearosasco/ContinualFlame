import pytest
import numpy as np
from contflame.dataset import SplitMNIST
from pathlib import Path
from itertools import chain, combinations

root = Path('./data')

def test_data_dir():
    ds = SplitMNIST(root=root)
    assert Path(root/'mnist-python').is_dir()

def test_iterate():
    ds = SplitMNIST(root=root)
    for x, y in iter(ds):
        pass

def test_classes1():
    t = range(5)
    subsets = chain.from_iterable(combinations(t, r) for r in range(1, len(t) + 1))
    for s in subsets:
        ds = SplitMNIST(root=root, classes=list(s))
        l = set([x[1] for x in iter(ds)])
        assert set(s) == l

def test_classes2():
    examples = [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]

    t = range(5)
    subsets = chain.from_iterable(combinations(t, r) for r in range(1, len(t) + 1))
    for s in subsets:
        sum = 0
        for i in s:
            sum += examples[i]

        ds = SplitMNIST(root=root, classes=list(s))
        assert len(ds) == sum

def test_split1():
    train = SplitMNIST(root=root, dset='train', valid=0.2, classes=[0, 1])
    valid = SplitMNIST(root=root, dset='valid', valid=0.2, classes=[0])

    for x in iter(train):
        for y in iter(valid):
            assert x == y


def test_split2():
    train = SplitMNIST(root=root, dset='train', valid=0.2, classes=[0])
    valid = SplitMNIST(root=root, dset='valid', valid=0.3, classes=[0])

    for x in iter(train):
        for y in iter(valid):
            if x != y:
                assert True
                return
    assert False
#
# def test_meta():
#     pass
#
# def test_transform():
#     pass
#
# def test():
#     ds = SplitMNIST(root=root, classes=[0])
#     print(set([x[1] for x in iter(ds)]))

test_split2()