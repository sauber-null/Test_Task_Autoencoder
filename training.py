from datetime import datetime
from os import path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

from inference import inference
from model import get_model
from perceptual_loss import VGGPerceptualLoss
from residual_model import get_residual
from unet_model import get_unet


def training(data_train, data_val):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    now = datetime.now()
    epoch_number = 5
    pixels_ratio = 0.00962
    model_path = './model.pth'
    model_name = 'ups_1.21'
    log_dir = './runs/' + model_name + '_' + now.strftime('%d%m%Y-%H%M%S')
    writer = SummaryWriter(log_dir=log_dir)

    # load train, val and test data and put it onto cuda
    train_loader = DataLoader(data_train, batch_size=8,
                              shuffle=True, num_workers=12, drop_last=True)

    val_loader = DataLoader(data_val, batch_size=8,
                            shuffle=True, num_workers=12, drop_last=True)

    # net = get_unet()
    net = get_model()

    # criterion = nn.L1Loss()
    criterion = nn.BCEWithLogitsLoss()
    # criterion = VGGPerceptualLoss()
    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pixels_ratio))

    net = net.to(device)
    criterion = criterion.to(device)
    # criterion1 = criterion1.to(device)
    # criterion2 = criterion2.to(device)

    # optimizer = optim.SGD(net.parameters(), lr=0.1)
    # optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.5)
    optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.7, 0.999))
    # optimizer = optim.RMSprop(net.parameters(), lr=0.0001, weight_decay=1e-4, momentum=0.9)

    # scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1000, gamma=0.1)
    # scheduler = optim.lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=0.0001, max_lr=0.1, step_size_up=1000)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min')
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=1000)

    print('Training started')

    for epoch in range(epoch_number):
        net.train()
        # get time for pictures naming
        now = datetime.now()
        loop = tqdm(train_loader)
        for i, data in enumerate(loop):
            src, label = data
            running_loss = 0.0
            # data to the device
            src, label = src.to(device), label.to(device)
            optimizer.zero_grad()

            output = net(src)
            loss = criterion(output, label)
            # use for the combined loss
            # loss = criterion1(output, label) + criterion2(output, label)

            loss.backward()

            optimizer.step()
            running_loss += loss.item()
            # pass data to the tensorboard
            writer.add_scalar('Training loss',
                loss.item(),
                epoch * len(train_loader) + i)
            # print out the training progress
            loop.set_description(f'Epoch [{epoch + 1}/{epoch_number}]')
            loop.set_postfix(loss=loss.item())
            writer.flush()
            # scheduler.step()

        # saving source, label and output as images in the 'log' folder
        save_image(src[:4], './log/{}_src_{}_epoch_{}.png'.format(now.strftime('%d%m%Y-%H%M%S'), model_name, epoch + 1))
        save_image(src[:4], './log/src/{}_src_{}_epoch_{}.png'.format(now.strftime('%d%m%Y-%H%M%S'), model_name, epoch + 1))

        save_image(label[:4], './log/{}_label_{}_epoch_{}.png'.format(now.strftime('%d%m%Y-%H%M%S'), model_name, epoch + 1))
        save_image(label[:4], './log/label/{}_label_{}_epoch_{}.png'.format(now.strftime('%d%m%Y-%H%M%S'), model_name, epoch + 1))

        save_image(torch.sigmoid(output[:4]).float().cpu(), './log/{}_output_{}_epoch_{}.png'.format(now.strftime('%d%m%Y-%H%M%S'), model_name, epoch + 1))
        save_image(torch.sigmoid(output[:4]).float().cpu(), './log/output/{}_output_{}_epoch_{}.png'.format(now.strftime('%d%m%Y-%H%M%S'), model_name, epoch + 1))

        # run validation
        with torch.no_grad():
            net.eval()
            loop = tqdm(val_loader)
            running_loss = 0.0
            for i, data in enumerate(loop):
                src, label = data
                src, label = src.to(device), label.to(device)

                output = net(src)
                loss = criterion(output, label)
                # use for the combined loss
                # loss = criterion1(output, label) + criterion2(output, label)
                running_loss += loss.item()
                # pass data to the tensorboard
                writer.add_scalar('Validation loss',
                    loss.item(),
                    epoch * len(val_loader) + i)
                # print out the validation progress
                loop.set_description(f'Validation Epoch [{epoch + 1}/{epoch_number}]')
                loop.set_postfix(loss=loss.item())
                # scheduler.step()
        
        # saving val source, label and output as images in the 'log' folder
        save_image(src[:4], './log/{}_src_val_{}_epoch_{}.png'.format(now.strftime('%d%m%Y-%H%M%S'), model_name, epoch + 1))
        save_image(src[:4], './log/src_val/{}_src_val_{}_epoch_{}.png'.format(now.strftime('%d%m%Y-%H%M%S'), model_name, epoch + 1))

        save_image(label[:4], './log/{}_label_val_{}_epoch_{}.png'.format(now.strftime('%d%m%Y-%H%M%S'), model_name, epoch + 1))
        save_image(label[:4], './log/label_val/{}_label_val_{}_epoch_{}.png'.format(now.strftime('%d%m%Y-%H%M%S'), model_name, epoch + 1))

        save_image(torch.sigmoid(output[:4]).float().cpu(), './log/{}_output_val_{}_epoch_{}.png'.format(now.strftime('%d%m%Y-%H%M%S'), model_name, epoch + 1))
        save_image(torch.sigmoid(output[:4]).float().cpu(), './log/output_val/{}_output_val_{}_epoch_{}.png'.format(now.strftime('%d%m%Y-%H%M%S'), model_name, epoch + 1))

        # print images in tensorboard
        writer.add_images('Images/Source batch', src[:4], epoch + 1)
        writer.add_images('Images/Label batch', label[:4], epoch + 1)
        writer.add_images('Images/Output batch', torch.sigmoid(output[:4]).float().cpu(), epoch + 1)

    print('Training is finished')
    # save model
    torch.save(net.state_dict(), model_path)

    writer.close()
