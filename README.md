[![Build Status](https://travis-ci.com/andrew-r96/ContinualFlame.svg?branch=main)](https://travis-ci.com/andrew-r96/ContinualFlame)
# ContinualFlame
Small lightweight package for Continual Learning in PyTorch.
## Installation
For now the package is hosted on TestPyPi. To install it you just need to run:
```bash
pip install --index-url https://test.pypi.org/simple/ continual-flame
```
## Usage
To use the package you just need to import it inside your project.
```bash
import contflame as cf
```
At the moment the package contains just the dataset module.
# Dataset
This module contains datasets normally used in the continual learning scenario. The main ones are:
- SplitMNIST - contains the standard MNIST dataset but it lets you select the classes to use during training
# Examples
```python
from contflame.dataset import SplitMNIST

valid = []
for i in range(1, 10, 2)
  train_dataset = SplitMNIST(tasks=[i, i+1], dset='train', valid=0.2)
  valid.append(SplitMNIST(tasks=[i, i+1], dset='valid', valid=0.2))
  
  for e in epochs:
    # train the model on train_dataset
    # ...
    
  for v in valid:
    # test the model on the current and the previous tasks
    # ...
```
