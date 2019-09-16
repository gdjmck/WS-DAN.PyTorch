import logging
import warnings
import pickle
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
parser.add_option('--testset', type=str, help='directory of testset data')
parser.add_option('--sd', '--save-dir', dest='save_dir', default='./models',
                    help='saving directory of .ckpt models (default: ./models)')
parser.add_option('--cpu', dest='cpu', default=False,
                    help='use cpu only if turned on')

(options, args) = parser.parse_args()


def test():
    # Default Parameters
    beta = 1e-4
    theta_c = 0.5
    theta_d = 0.5
    crop_size = (256, 256)  # size of cropped images for 'See Better'
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
    feature_center = torch.zeros(num_classes, num_attentions, net.num_features * net.expansion).to(torch.device("cuda") if not options.cpu else torch.device('cpu'))

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
    if not options.cpu:
        cudnn.benchmark = True
        net.to(torch.device("cuda"))
        net = nn.DataParallel(net)

    ##################################
    # Load dataset
    ##################################   
    test_dataset = dataset.ImageFolderWithName(phase='val')

    result = {}
    net.eval()
    with torch.no_grad():
        for (x, y, fn) in iter(test_dataset):
            x = x.unsqueeze(0)
            if not options.cpu:
                x = x.to(torch.device('cuda'))

            ##################################
            # Raw Image
            ##################################
            y_pred, feature_matrix, attention_map, metric = net(x)
            result[fn] = metric.cpu().numpy()
            continue

            ##################################
            # Attention Cropping
            ##################################
            crop_mask = F.upsample_bilinear(attention_map, size=(x.size(2), x.size(3))) > theta_c
            crop_images = []
            for batch_index in range(crop_mask.size(0)):
                nonzero_indices = torch.nonzero(crop_mask[batch_index, 0, ...])
                height_min = nonzero_indices[:, 0].min()
                height_max = nonzero_indices[:, 0].max()
                width_min = nonzero_indices[:, 1].min()
                width_max = nonzero_indices[:, 1].max()
                crop_images.append(F.upsample_bilinear(x[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max], size=crop_size))
            crop_images = torch.cat(crop_images, dim=0)   
            y_crop_pred, _, _ = net(crop_images)  

            ##################################
            # Attention Dropping
            ##################################
            drop_mask = F.upsample_bilinear(attention_map, size=(x.size(2), x.size(3))) <= theta_d
            drop_images = x * drop_mask.float()         
            y_drop_pred, _, _ = net(drop_images)   
            _, y_pred = y_pred.topk(1, 1, True, True)
            _, y_crop_pred = y_crop_pred.topk(1, 1, True, True)
            _, y_drop_pred = y_drop_pred.topk(1, 1, True, True)
            result[y] = {'pred': y_pred.cpu().numpy()[0, 0],
                         'pred_crop': y_crop_pred.cpu().numpy()[0, 0],
                         'pred_drop': y_drop_pred.cpu().numpy()[0, 0]}
    
    with open('result.pkl', 'wb') as f:
        pickle.dump(result, f)

if __name__ == '__main__':
    dataset.config['datapath'] = options.testset
    test()
