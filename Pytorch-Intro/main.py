import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sn
import itertools

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from sklearn.metrics import confusion_matrix


class MyLeNet5(nn.Module):

    def __init__(self):
        super(MyLeNet5, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
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
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def initiate_loss():
    # Creating initial batch
    loss_fn = torch.nn.CrossEntropyLoss()

    dummy_outputs = torch.randn(1, 10)
    dummy_labels = dummy_outputs.view(1, -1)

    print(dummy_outputs)
    print(dummy_labels)

    loss = loss_fn(dummy_outputs, dummy_labels)
    print('Total loss for this batch: {:.4f}'.format(loss.item()))

    return loss_fn


def train_single_epoch(model, loss_function, trainloader, device, optimizer, batch_size):

    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(trainloader):

        # Every data instance is an input + label pair
        inputs, labels = data

        # If GPU is available
        if str(device) == 'cuda':
            inputs, labels = inputs.cuda(), labels.cuda()  # GPU

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_function(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 1:
            last_loss = running_loss / batch_size
            print("\rbatch {}/{}, loss: {:.4f}".format(i + 1, len(trainloader), last_loss, end='', flush=True))
            running_loss = 0.

    return last_loss


def train_nn(model, loss_function, total_epochs, trainloader, device, optimizer, batch_size):

    best_loss = 1_000_000.

    for epoch in range(total_epochs):

        print('Epoch: ', epoch+1)

        model.train(True)

        avg_loss = train_single_epoch(model, loss_function, trainloader, device, optimizer, batch_size)

        model.train(False)

        running_vloss = 0.0
        print('Validation in process...')
        j = 0
        for i, vdata in enumerate(trainloader):
            vinputs, vlabels = vdata
            # If GPU is available
            if str(device) == 'cuda':
                vinputs, vlabels = vinputs.cuda(), vlabels.cuda()  # GPU
            voutputs = model(vinputs)
            vloss = loss_function(voutputs, vlabels)
            running_vloss += vloss

            if j == 1000:
                break

            j += 1

        avg_vloss = running_vloss / batch_size
        print('LOSS train {:.4f} valid {:.4f}'.format(avg_loss, avg_vloss))

        if avg_vloss < best_loss:
            best_loss = avg_vloss


def run_test(test_loader, device, model):
    y_pred = []
    y_true = []

    for i, vdata in enumerate(test_loader):
        vinputs, vlabels = vdata
        # If GPU is available
        if str(device) == 'cuda':
            vinputs, vlabels = vinputs.cuda(), vlabels.cuda()  # GPU
        voutputs = model(vinputs)
        output = torch.max(torch.exp(voutputs), 1)[1]
        y_pred.extend(output.to('cpu').numpy())
        y_true.extend(vlabels.to('cpu').numpy())

        if i % 500 == 0:
            print("\rTest batch {}/{}".format(i, len(test_loader), end='', flush=True))

    return y_pred, y_true


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

    print('1: {}, 2: {}, 3: {}'.format(
        autograd_variable.grad_fn, autograd_variable2.grad_fn, autograd_variable3.grad_fn))

    # Neural Networks
    print('\n\nBuilding of a simple LetNet5:')

    model = MyLeNet5()
    print(model)

    params = list(model.parameters())
    print('Amount of parameters: ', len(params))
    print(params[0].size())

    print('Initiate loss:')
    loss_function = initiate_loss()
    print(loss_function)

    print('Initiate optimizer:')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    print(optimizer)

    print('Initiate data:')
    data = torch.randn(1, 3, 32, 32)
    print(data)

    print('Testing randon data on untrained model:')
    print(model(data))

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # functions to show an image

    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    # initiate training
    if str(device) == 'cuda':
        model.to(device)  # GPU

    print('Initiate training:')
    total_epochs = 10

    print(total_epochs)
    train_nn(model, loss_function, total_epochs, trainloader, device, optimizer, batch_size)

    print('Test:')
    y_pred, y_true = run_test(testloader, device, model)

    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1), index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.show()


if __name__ == '__main__':
    main()
