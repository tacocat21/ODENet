from model import OdeNet224
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch

def run_model():
    model = OdeNet224('res', 1, 10, 3, 64)
    # transform = transforms.Compose([
    #     transforms.Resize(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.4467, 0.4398, 0.4066], [0.2185, 0.2159, 0.2183])
    # ])
    # d = datasets.STL10(root='.data/stl10', split='train', download=True, transform=transform)
    x, y = torch.load('test/test_1')
    y = model(x.view(1, 3, 224, 224))
    return y

if __name__ == '__main__':
    # cProfile.run('code_profile.run_model()', 'profile_result')
    run_model()