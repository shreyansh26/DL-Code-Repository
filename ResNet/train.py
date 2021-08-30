from typing import List, Optional
from torch import nn
from torch.autograd.variable import Variable
from model import ResNetBase

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

CUDA_AVAILABLE = torch.cuda.is_available()

class Configs():
    n_blocks: List[int] = [6, 6, 6]
    n_channels: List[int] = [16, 32, 64]
    bottlenecks: Optional[List[int]] = None
    first_kernel_size: int = 3
    bottlenecks: List[int] = [8, 16, 16]
    device = 'cuda'


def get_model(c: Configs):
    base = ResNetBase(c.n_blocks, c.n_channels, c.bottlenecks, img_channels=3, first_kernel_size=c.first_kernel_size)
    classification = nn.Linear(c.n_channels[-1], 10)
    model = nn.Sequential(base, classification)

    return model.to(c.device)

def main():
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 128

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
    #                                         shuffle=False, num_workers=2)

    # classes = ('plane', 'car', 'bird', 'cat',
    #         'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    c = Configs()
    model = get_model(c)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2.5e-4)

    Tensor = torch.cuda.FloatTensor if CUDA_AVAILABLE else torch.FloatTensor


    for epoch in range(500):  # loop over the dataset multiple times

        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            inputs = Variable(inputs.type(Tensor))
            labels = Variable(labels.type(Tensor))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    PATH = './cifar_net.pth'
    torch.save(model.state_dict(), PATH)

if __name__ == "__main__":
    main()