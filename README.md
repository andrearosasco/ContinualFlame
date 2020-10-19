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
**SplitMNIST**

In the following example the training tasks are five binary classification tasks on subsequent pairs of digit (i.e task 1 (0, 1), task 2 (2, 3), ...)
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
**PermutedMNIST**

To get a random permutation set tile to (1, 1). The same random permutation, selected by the task id, will be applied to all the data points.

```python
PermutedMNIST(tile=(1, 1), task=1)
```
<img style="float: right;" src="https://user-images.githubusercontent.com/47559809/96425928-da7d2f00-11fc-11eb-95d9-8035dde0e333.png" width="96">
You can also apply the permutation row (or column) wise by setting the corresponding dimension of the tile equal to the one of the image

```python
PermutedMNIST(tile=(1, 28), task=1)
```
<img style="float: right;" src="https://user-images.githubusercontent.com/47559809/96428444-0e0d8880-1200-11eb-814b-376496129f63.png" width="96">
Or try to maintain high level spatial feature by setting a bigger tile.

```python
PermutedMNIST(tile=(8, 8), task=1)
```
<img style="float: right;" src="https://user-images.githubusercontent.com/47559809/96429174-ef5bc180-1200-11eb-81bf-3bbb1dd6a515.png" width="96">

To get the images without any permutation set the tile to (28, 28) (default value).
