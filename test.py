import logging
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from optparse import OptionParser

from utils import accuracy
from models import *
#from dataset import *
import dataset

parser = OptionParser()
parser.add_option('-c', '--ckpt', dest='ckpt', default=False,
                    help='load checkpoint model (default: False)')
parser.add_option('-v', '--verbose', dest='verbose', default=100, type='int',
                    help='show information for each <verbose> iterations (default: 100)')
parser.add_option('--testset', required=True, type=str, help='directory of testset data')
parser.add_option('--sd', '--save-dir', dest='save_dir', default='./models',
                    help='saving directory of .ckpt models (default: ./models)')

(options, args) = parser.parse_args()


def test():
    ##################################
    # Initialize model
    ##################################
    image_size = 512
    num_classes = 98
    num_attentions = 32
    start_epoch = 0

    feature_net = inception_v3(pretrained=True)
    net = WSDAN(num_classes=num_classes, M=num_attentions, net=feature_net)
    
    # feature_center: size of (#classes, #attention_maps, #channel_features)
    feature_center = torch.zeros(num_classes, num_attentions, net.num_features * net.expansion).to(torch.device("cuda"))

    ckpt = options.ckpt

    # Load ckpt and get state_dict
    checkpoint = torch.load(ckpt)
    state_dict = checkpoint['state_dict']

    # Load weights
    net.load_state_dict(state_dict)

    # load feature center
    if 'feature_center' in checkpoint:
        feature_center = checkpoint['feature_center'].to(torch.device("cuda"))

    ##################################
    # Use cuda
    ##################################
    cudnn.benchmark = True
    net.to(torch.device("cuda"))
    net = nn.DataParallel(net)

    ##################################
    # Load dataset
    ##################################   
    test_dataset = CustomDataset(phase='test')

    for (x, y) in iter(test_dataset):
        print(x.shape, y)

if __name__ == '__main__':
    dataset.config['datapath'] = options.testset
    test()