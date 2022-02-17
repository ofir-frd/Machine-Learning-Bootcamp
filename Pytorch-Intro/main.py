

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():

    # PyTorch Tensors:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    a = torch.rand(2, 2).to(device)
    b = torch.rand(2, 2).to(device)

    c_constant = [[1, 2], [3, 4]]
    c_constant = torch.tensor(c_constant).to(device)

    a1 = a + c_constant
    b1 = b + c_constant

    a2 = a1 * c_constant
    b2 = b1 * c_constant

    a2b2 = a2 + b2

    print('constant: {}, a: {}, a1: {}, a2: {}, b: {}, b1: {}, b2: {}, a2b2: {}'.format(
        c_constant, a, a1, a2, b, b1, b2, a2b2))

    a2b2 = a2b2.to('cpu').numpy()
    print(type(a2b2), a2b2)

    a2b2 = torch.from_numpy(a2b2)
    print(type(a2b2), a2b2)

    print('View: ', a2b2.size())

    # AutoGrad:
    autograd_variable = Variable(torch.rand(10), requires_grad=True)
    print('1: ', autograd_variable)

    autograd_variable2 = autograd_variable + torch.tensor(2)
    print('2: ', autograd_variable2)

    autograd_variable3 = torch.mean(autograd_variable2)
    print('3: ', autograd_variable3)

    autograd_variable3.backward()

    print('\n\n1: {}, 2: {}, 3: {}'.format(autograd_variable.grad_fn, autograd_variable2.grad_fn, autograd_variable3.grad_fn))

    # Neural Networks



if __name__ == '__main__':
    main()
