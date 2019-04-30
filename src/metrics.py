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



if __name__ == '__main__':
    #transform_train = transforms.Compose([
    #    transforms.Resize(224),
    #    transforms.ToTensor(),
    #])
    model = OdeNet224('res', 0.001, 10, 3, 64)

    #dataloader = DataLoader(datasets.STL10(root='.data/stl10', split='train', download=True, transform=transform_train), batch_size=64)
    img, label = torch.load('single_test')
    if len(img.shape) < 4:
        img = img.view(1, 3, 224,224)
    print(img.shape)
    ram_used = measure_function_difference(get_current_ram_used, forward, (model, img))
    print(ram_used)
    # ram_used = measure_function_difference(get_current_ram_used, OdeNet, ('squeeze', 0.001, 10, 3, 64))
    # print('ODENet (squeeze downsampling) ram used = {} bytes'.format(ram_used))
    # ram_used = measure_function_difference(get_current_ram_used, torchvision.models.resnet18, (False,))
    # print('Resnet ram used = {} bytes'.format(ram_used))
    ram_used = measure_function_difference(get_current_ram_used, torchvision.models.squeezenet1_1, (False,))
    print('SqueezeNet ram used = {} bytes'.format(ram_used))
    # measure_model_ram_memory()
    # print(sum_model_parameter_size(OdeNet('conv', 0.001, 10, 3)))

