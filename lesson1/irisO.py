import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
xtrain, xtest, ytrain, ytest = train_test_split(
    iris.data, iris.target, test_size=0.5)

xtrain = torch.from_numpy(xtrain).type('torch.FloatTensor')
ytrain = torch.from_numpy(ytrain).type('torch.FloatTensor')
xtest = torch.from_numpy(xtest).type('torch.FloatTensor')
ytest = torch.from_numpy(ytest).type('torch.FloatTensor')