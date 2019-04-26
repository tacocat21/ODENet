# most code from https://github.com/rtqichen/torchdiffeq/

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


from torchdiffeq import odeint_adjoint as odeint
import itertools

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut


class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc, tolerance):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()
        self.tolerance = tolerance

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=self.tolerance, atol=self.tolerance)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val



def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()



class OdeNet(nn.Module):
    def __init__(self, downsampling_method, tolerance, num_classes, num_in_channels, hidden_channels=64):
        """
        Images must be num_in_channels x 32 x 32

        :param downsampling_method:
        :param tolerance:
        :param num_classes:
        :param num_in_channels:
        """
        super(OdeNet, self).__init__()
        if downsampling_method == 'conv':
            self.downsampling_layers = [
                nn.Conv2d(num_in_channels, hidden_channels, 3, 1),
                norm(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, hidden_channels, 4, 2, 1),
                norm(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, hidden_channels, 4, 2, 1),
            ]
        elif downsampling_method == 'res':
            self.downsampling_layers = [
                nn.Conv2d(num_in_channels, 64, 3, 1),
                ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
                ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
            ]
        elif downsampling_method == 'squeeze':
            self.downsampling_layers = [
                nn.Conv2d(num_in_channels, hidden_channels, 1, 1),
                norm(hidden_channels),
                nn.ReLU(inplace=True)
            ]
        else:
            raise RuntimeError('downsampling_method must be conv or res')
        self.tolerance = tolerance
        self.feature_layers = [ODEBlock(ODEfunc(hidden_channels), self.tolerance)]
        self.fc_layers = [norm(hidden_channels), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(hidden_channels, num_classes)]
        self.seq = nn.Sequential(*self.downsampling_layers, *self.feature_layers, *self.fc_layers)
        if torch.cuda.is_available():
            self.cuda()

    def parameters(self, recurse=True):
        return self.seq.parameters()

    def forward(self, x):
        # y = self.downsampling_layers(input)
        # y = self.feature_layers(y)
        # y = y.view(input.size(0), -1)
        # y = self.fc_layers(y)
        # return y
        return self.seq(x)



