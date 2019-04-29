# code to train models I'm comparing ODEnet with
import argparse
import torchvision

def get_squeezenet(num_classes):
    model = torchvision.models.SqueezeNet(version=1.1, num_classes=num_classes)
    return model

def train_model(model, train_loader, test_loader):
    pass

def parse():
    parser = argparse.ArgumentParser(description='Train a network on stl10 dataset')
    parser.add_argument('--model', type=str, choices=['squeezenet', 'resnet', ], required=True)

    parser.add_argument('--save', type=str, default='./experiment1')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--nepochs', type=int, default=300)
    parser.add_argument('--tol', type=float, default=1e-3)
    return parser


if __name__ == '__main__':
    parser = parse()
    args = parser.parse_args()

    optimizer_type = args.optimizer
    batch_size = args.batch_size
    test_batch_size = 250
    lr = args.lr
    num_epochs = args.nepochs

