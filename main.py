import argparse
import random

import numpy as np
import torch

from dataset import get_data
from inference import inference
from training import training


def main():
    manualSeed = 42
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    # cudnn.enabled = False 
    # cudnn.benchmark = False
    # cudnn.deterministic = True

    arg_parser = argparse.ArgumentParser(description='Run mode: train/val or inference')
    arg_parser.add_argument('--mode', type=str, help='Input \'train\' or \'inference\'')
    args = arg_parser.parse_args()

    data_train, data_val, data_test = get_data()

    if args.mode == 'train':
        training(data_train, data_val)
    elif args.mode == 'inference':
        inference(data_test, is_custom=True, is_cpu=False)
    else:
        print('You have inputted the wrong mode')


if __name__ == '__main__':
    main()
