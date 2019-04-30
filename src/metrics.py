import sys
import psutil
import os
import torch
import torchvision
import gc
from model import OdeNet, OdeNet224
import ipdb
import skimage
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import time

class Metric:
    def __init__(self):
        pass


def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())


def cpuStats():
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory GB:', memoryUse)

def get_current_ram_used():
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]
    return memoryUse

# cpuStats()
# memReport()

def measure_function_difference(measurement_function, function_measured, function_args):
    # ipdb.set_trace()
    load = nn.Conv2d(3, 64, 5, 1) # use to load pytorch before measuring
    load2 = torch.Tensor([1])
    if torch.cuda.is_available():
        load.cuda()
        load2.cuda()
    before = measurement_function()
    model  = function_measured(*function_args)
    # print(model)
    # model = OdeNet('conv', tolerance=0.001, num_classes=10, num_in_channels=3)
    after = measurement_function()
    del model
    difference = after- before
    # print("model size = {} bytes".format(model_size))
    return difference



def sum_model_parameter_size(model):
    total = 0
    for p in model.parameters():
        total += sys.getsizeof(p)
    return total

def forward(model, img):
    res = model(img)
    return res

def measure_odenet_224():
    ram_used = measure_function_difference(get_current_ram_used, OdeNet224, ('conv', 0.001, 10, 3, 64))
    print('ODENet (conv downsampling) ram used = {} bytes'.format(ram_used))

def run_n_times(model, img, n):
    for i in range(n):
        y = model(img)
    return y

def parse():
    parser = argparse.ArgumentParser(description='Measure memory or time to run network')
    parser.add_argument('--metric', type=str, default='memory', choices=['memory-model', 'memory-inference', 'time'], required=True)
    parser.add_argument('--batch-size', type=int, default=1, choices=[1, 10, 32])
    parser.add_argument('--model', type=str, default='ode', choices=['ode', 'squeeze'])
    parser.add_argument('--downsampling-method', type=str, default='conv', choices=['conv', 'res'])
    parser.add_argument('--tol', type=float, default=1e-3)

    return parser

if __name__ == '__main__':
    parser = parse()
    args = parser.parse_args()
    if args.model == 'ode' and args.metric != 'memory-model':
        model = OdeNet224(args.downsampling_method, args.tol, 10, 3, 64)
        model.eval()
    elif args.model == 'squeeze' and args.metric != 'memory-model':
        model = torchvision.models.squeezenet1_1(False)
        model.eval()
    if args.batch_size == 1:
        img, label = torch.load('test/test_1')
        img = img.view(1, 3, 224,224)
    elif args.batch_size == 10:
        img, label = torch.load('test/test_10')
    elif args.batch_size == 32:
        img, label = torch.load('test/test_32')
    print(img.shape)

    # load pytorch overhead
    preload = torch.tensor([1])
    preload2 = torch.tensor([3])
    preload = preload + preload2

    with torch.no_grad():
        if args.metric == 'memory-inference':
            test = model(img) # run once to load the model
            del test
            ram_used = measure_function_difference(get_current_ram_used, forward, (model, img))
            print("model {} used {} bytes to run {} images".format(args.model, ram_used, args.batch_size))
        elif args.metric == 'time':
            n = 25
            test = model(img) # run once to load the model
            del test
            _time = measure_function_difference(time.time,run_n_times, (model, img, n))
            print("model {} ran {} inferences in {}s. Avg time = {}s. Each batch has {} images".format(
                args.model, n, _time, _time/n, args.batch_size))
        elif args.metric == 'memory-model':
            ram_used = measure_function_difference(get_current_ram_used, OdeNet, ('squeeze', 0.001, 10, 3, 64))
            print('ODENet (squeeze downsampling) ram used = {} bytes'.format(ram_used))
    # ram_used = measure_function_difference(get_current_ram_used, torchvision.models.resnet18, (False,))
    # print('Resnet ram used = {} bytes'.format(ram_used))
    # ram_used = measure_function_difference(get_current_ram_used, torchvision.models.squeezenet1_1, (False,))
    # print('SqueezeNet ram used = {} bytes'.format(ram_used))
    # measure_model_ram_memory()
    # print(sum_model_parameter_size(OdeNet('conv', 0.001, 10, 3)))

