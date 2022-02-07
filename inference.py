from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import get_model
from unet_model import get_unet


def load_model(device):
    model_path = './model.pth'
    net = get_model()
    net.load_state_dict(torch.load(model_path))
    net = net.to(device)
    return net


def load_unet(device):
    unet_model_path = './unet_best.pth'
    net = get_unet()
    net.load_state_dict(torch.load(unet_model_path))
    net = net.to(device)
    return net


def inference(data_test, is_custom: bool = True, is_cpu: bool = False):
    """
    This function takes the data and runs the model to get a prediction for the input data

    Parameters
    ----------
    data_test
        Data to make the inference on
    is_custom : bool
        Flag whether to run a custom model (if True) or a U-Net (if False), default is True
    is_cpu : bool
        Flag whether to run program on CPU (if True) or on GPU (if False), default is False
    ----------
    """
    
    if is_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if is_custom:
        net = load_model(device)
    else:
        net = load_unet(device)

    test_loader = DataLoader(data_test, batch_size=1,
                             shuffle=True, num_workers=12, drop_last=True)

    start_time = datetime.now()

    net.eval()
    with torch.no_grad():
        loop = tqdm(test_loader)
        for i, (src, label) in enumerate(loop):
            src, label = src.to(device), label.to(device)
            output = net(src)
            if i == 1000:
                break
        end_time = datetime.now()
        print('Inference for 1000 images took ', end_time-start_time)
    

if __name__ == '__main__':
    inference()
