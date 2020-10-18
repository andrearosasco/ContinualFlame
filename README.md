[![Build Status](https://travis-ci.com/andrew-r96/ContinualFlame.svg?branch=main)](https://travis-ci.com/andrew-r96/ContinualFlame)
# ContinualFlame
Small lightweight package for Continual Learning in PyTorch.
## Installation
For now the package is hosted on TestPyPi. To install it you just need to run:
```bash
pip install continual-flame
```
## Usage
To use the package you just need to import it inside your project.
```python
import contflame as cf
```
At the moment the package contains just the dataset module.
# Dataset
This module contains datasets normally used in the continual learning scenario. The main ones are:
- SplitMNIST - MNIST dataset split in classes. It allows to create different subtasks by including custom subsets of classes.
- PermutedMNIST - permuted MNIST dataset. It allows to choose the shape of the applied permutation.
- SplitCIFAR100
- PermutedCIFAR100
# Examples
SplitMNIST
```python
from cont_flame.dataset import SplitMNIST

valid = []
for i in range(1, 10, 2)
  train_dataset = SplitMNIST(classes=[i, i+1], dset='train', valid=0.2)
  valid.append(SplitMNIST(classes=[i, i+1], dset='valid', valid=0.2))
  
  for e in epochs:
    # train the model on train_dataset
    # ...
    
  for v in valid:
    # test the model on the current and the previous tasks
    # ...
```
PermutedMNIST
