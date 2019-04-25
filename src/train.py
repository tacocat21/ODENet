from os import makedirs
import ipdb
import os
import logging
import time
import numpy as np
from model import OdeNet
import torch
import torch.nn as nn
import dataset

def learning_rate_with_decay(lr, batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate =  lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader):
    with torch.no_grad():
        total_correct = 0
        for x, y in dataset_loader:
            if torch.cuda.is_available():
                x = x.cuda()
            y = one_hot(np.array(y.numpy()), 10)
            target_class = np.argmax(y, axis=1)
            predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
            total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger

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



def train_odenet(model, train_loader, train_eval_loader, test_loader, num_epochs, batch_size, lr, logger, save_dir, val_break_threshold=0.98):

    logger.info(model)
    logger.info('Number of parameters: {}'.format(count_parameters(model)))

    criterion = nn.CrossEntropyLoss()
    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    lr_fn = learning_rate_with_decay(
        lr, batch_size, batch_denom=128, batches_per_epoch=batches_per_epoch, boundary_epochs=[60, 100, 140],
        decay_rates=[1, 0.1, 0.01, 0.001]
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    best_acc = 0
    batch_time_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()
    end = time.time()

    for itr in range(num_epochs * batches_per_epoch):

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_fn(itr)

        optimizer.zero_grad()
        x, y = data_gen.__next__()
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        logits = model(x)
        loss = criterion(logits, y)

        # forward pass
        nfe_forward = model.feature_layers[0].nfe
        model.feature_layers[0].nfe = 0

        loss.backward()
        optimizer.step()

        # backward pass
        nfe_backward = model.feature_layers[0].nfe
        model.feature_layers[0].nfe = 0

        batch_time_meter.update(time.time() - end)

        f_nfe_meter.update(nfe_forward)
        b_nfe_meter.update(nfe_backward)
        end = time.time()

        if itr % batches_per_epoch == 0:
            with torch.no_grad():
                train_acc = accuracy(model, train_eval_loader)
                val_acc = accuracy(model, test_loader)
                if val_acc > best_acc:
                    torch.save({'state_dict': model.state_dict()}, os.path.join(save_dir, 'model.pth'))
                    best_acc = val_acc
                logger.info(
                    "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
                    "Train Acc {:.4f} | Test Acc {:.4f}".format(
                        itr // batches_per_epoch, batch_time_meter.val, batch_time_meter.avg, f_nfe_meter.avg,
                        b_nfe_meter.avg, train_acc, val_acc
                    )
                )
                if val_acc > val_break_threshold:
                    logger.info("Ending training early. Validation accuracy = {}".format(val_acc))
                    return


if __name__ == '__main__':


    dataset_name = 'cifar10'
    # dataset_name = 'mnist'
    save_dir = './cache/{}'.format(dataset_name)
    logpath =  os.path.join(save_dir, 'logs')
    if os.path.exists(logpath):
        os.remove(logpath)

    batch_size = 512
    test_batch_size = 1000
    lr = 0.1
    num_epochs = 180
    train_batch_size = 128
    

    if dataset_name == 'mnist':
        num_classes = 10
        num_in_channels = 1
        train_loader, test_loader, train_eval_loader = dataset.get_mnist_loaders(
            True, batch_size, test_batch_size
        )
        val_break_threshold = 0.99
    elif dataset_name == 'cifar10':
        num_classes = 10
        num_in_channels = 3
        train_loader, test_loader, train_eval_loader = dataset.get_cifar_10(
            True, batch_size, test_batch_size
        )
        val_break_threshold = 0.95
    makedirs(save_dir)
    logger = get_logger(logpath=logpath, filepath=os.path.abspath(__file__))


    model = OdeNet('conv', tolerance=0.001, num_classes=num_classes, num_in_channels=num_in_channels)

    logger.info(model)
    logger.info('Number of parameters: {}'.format(count_parameters(model)))

    criterion = nn.CrossEntropyLoss()


    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    train_odenet(model=model, train_loader=train_loader, train_eval_loader=train_eval_loader, test_loader=test_loader,
                 num_epochs=num_epochs, batch_size=train_batch_size, lr=lr, logger=logger, save_dir=save_dir, val_break_threshold=val_break_threshold)

