"""TRAINING
Created: May 04,2019 - Yuchong Gu
Revised: May 07,2019 - Yuchong Gu
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import time
import logging
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from optparse import OptionParser
from tensorboardX import SummaryWriter
from torchsummary import summary

from utils import accuracy, MetricLoss, plot_grad_flow_v2, center_loss
from models import *
from dataset import *

writer = None
step = 0

def main():
    parser = OptionParser()
    parser.add_option('-j', '--workers', dest='workers', default=16, type='int',
                      help='number of data loading workers (default: 16)')
    parser.add_option('-d', '--dim', dest='metric_dim', default=512, type='int',
                      help='dimension of metric')
    parser.add_option('-e', '--epochs', dest='epochs', default=80, type='int',
                      help='number of epochs (default: 80)')
    parser.add_option('-b', '--batch-size', dest='batch_size', default=16, type='int',
                      help='batch size (default: 16)')
    parser.add_option('-k', '--batch-k', dest='batch_k', default=4, type='int',
                      help='number of samples from same class (default: 4)')
    parser.add_option('-l', '--epoch-size', dest='sampler_len', default=2000, type='int',
                      help='batches in an epoch')
    parser.add_option('-c', '--ckpt', dest='ckpt', default=False,
                      help='load checkpoint model (default: False)')
    parser.add_option('-v', '--verbose', dest='verbose', default=100, type='int',
                      help='show information for each <verbose> iterations (default: 100)')

    parser.add_option('--lr', '--learning-rate', dest='lr', default=1e-3, type='float',
                      help='learning rate (default: 1e-3)')
    parser.add_option('--sf', '--save-freq', dest='save_freq', default=1, type='int',
                      help='saving frequency of .ckpt models (default: 1)')
    parser.add_option('--sd', '--save-dir', dest='save_dir', default='./models',
                      help='saving directory of .ckpt models (default: ./models)')
    parser.add_option('--init', '--initial-training', dest='initial_training', default=1, type='int',
                      help='train from 1-beginning or 0-resume training (default: 1)')
    parser.add_option('--freeze', '--freeze-feature', dest='freeze', default=False, action='store_true',
                      help='whether freeze feature extraction layers or not')
    parser.add_option('--raw', dest='raw', default=False, action='store_true',
                      help='whether just train raw image or using attention to augment data.')
    parser.add_option('--tb', '--tensorboard', dest='tb', default='./runs', 
                      help='tensorboard saving folder')

    global options
    (options, args) = parser.parse_args()

    logging.basicConfig(format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s', level=logging.INFO)
    warnings.filterwarnings("ignore")
    global writer
    writer = SummaryWriter(options.tb)

    ##################################
    # Initialize model
    ##################################
    if options.raw:
        logging.info('Training with raw images only')
    else:
        logging.info('Training with 3 process')
    global image_size
    image_size = 512
    num_classes = 98*2
    num_attentions = 32
    start_epoch = 0

    feature_net = inception_v3(pretrained=True)
    net = WSDAN(num_classes=num_classes, M=num_attentions, net=feature_net, metric_dim=options.metric_dim)

    # feature_center: size of (#classes, #attention_maps, #channel_features)
    feature_center = torch.zeros(num_classes, net.num_features * net.expansion).to(torch.device("cuda"))

    if options.ckpt:
        ckpt = options.ckpt

        if options.initial_training == 0:
            # Get Name (epoch)
            epoch_name = (ckpt.split('/')[-1]).split('.')[0]
            start_epoch = int(epoch_name)

        # Load ckpt and get state_dict
        checkpoint = torch.load(ckpt)
        state_dict = checkpoint['state_dict']

        # Load weights
        net.load_state_dict(state_dict)
        logging.info('Network loaded from {}'.format(options.ckpt))

        # load feature center
        if 'feature_center' in checkpoint:
            feature_center = checkpoint['feature_center'].to(torch.device("cuda"))
            logging.info('feature_center loaded from {}'.format(options.ckpt))

    ##################################
    # Initialize saving directory
    ##################################
    save_dir = options.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ##################################
    # Use cuda
    ##################################
    cudnn.benchmark = True
    net.to(torch.device("cuda"))
    summary(net, (3, 512, 512))

    ##################################
    # Load dataset
    ##################################
    train_dataset, validate_dataset = ImageFolderWithName(phase='train', shape=image_size), \
                                      ImageFolderWithName(phase='val'  , shape=image_size)

    train_loader, validate_loader = DataLoader(train_dataset, batch_sampler=CustomSampler(train_dataset, batch_size=options.batch_size, batch_k=options.batch_k, len=options.sampler_len),
                                               num_workers=options.workers, pin_memory=True), \
                                    DataLoader(validate_dataset, batch_sampler=CustomSampler(validate_dataset, batch_size=options.batch_size, batch_k=options.batch_k, len=options.sampler_len),
                                               num_workers=options.workers, pin_memory=True)

    train_params = net.parameters()
    optimizer = torch.optim.SGD(train_params, lr=options.lr, momentum=0.9, weight_decay=0.00001)
    loss = nn.CrossEntropyLoss()
    loss_metric = MetricLoss(options.batch_k)

    ##################################
    # Learning rate scheduling
    ##################################
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    ##################################
    # TRAINING
    ##################################
    logging.info('')
    logging.info('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
                 format(options.epochs, options.batch_size, len(train_dataset), len(validate_dataset)))

    for epoch in range(start_epoch, options.epochs):
        train(epoch=epoch,
              data_loader=train_loader,
              net=net,
              feature_center=feature_center,
              loss=loss,
              loss_metric=loss_metric,
              optimizer=optimizer,
              save_freq=options.save_freq,
              save_dir=options.save_dir,
              verbose=options.verbose)
        '''
        if not options.freeze:
            val_loss = validate(data_loader=validate_loader,
                                net=net,
                                loss=loss,
                                verbose=options.verbose)
        '''                      
        scheduler.step()

def train(**kwargs):
    # Retrieve training configuration
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    loss = kwargs['loss']
    loss_metric = kwargs['loss_metric']
    optimizer = kwargs['optimizer']
    feature_center = kwargs['feature_center']
    epoch = kwargs['epoch']
    save_freq = kwargs['save_freq']
    save_dir = kwargs['save_dir']
    verbose = kwargs['verbose']
    global step

    # Attention Regularization: LA Loss
    l2_loss = nn.MSELoss()

    # Default Parameters
    beta = 0.05
    theta_c = 0.5
    theta_d = 0.5
    crop_size = (256, 256)  # size of cropped images for 'See Better'

    # metrics initialization
    batches = 0
    epoch_loss_raw = np.array([0, 0, 0, 0, 0], dtype='float')  # loss_sum(classify, center, homo, heter)
    epoch_loss_crop = np.array([0, 0, 0, 0], dtype='float') # loss_sum(classify, homo, heter)
    epoch_loss_drop = np.array([0], dtype='float')
    epoch_acc = np.array([[0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0]], dtype='float')  # Top-1/3/5 Accuracy for Raw/Crop/Drop Images

    # begin training
    start_time = time.time()
    logging.info('Epoch %03d, Learning Rate %g' % (epoch + 1, optimizer.param_groups[0]['lr']))
    net.train()
    for i, (X, y) in enumerate(data_loader):
        batch_start = time.time()

        # obtain data for training
        X = X.to(torch.device("cuda"))
        y = y.to(torch.device("cuda"))

        ##################################
        # Raw Image
        ##################################
        y_pred, embeddings, attention_map = net(X)
        #print('RAW y_pred:\n', y_pred[0, ...], '\n', y_pred[2, ...])

        # loss
        metric_loss = loss_metric(embeddings)
        #if not options.freeze:
        loss_classify = loss(y_pred, y)
        loss_center = center_loss(embeddings, feature_center[y])
        batch_loss = loss_classify + loss_center + metric_loss[0] + metric_loss[1]
        epoch_loss_raw[0] += batch_loss.item()
        epoch_loss_raw[1] += loss_classify.item()
        epoch_loss_raw[2] += loss_center.item()
        epoch_loss_raw[3] += metric_loss[0].item()
        epoch_loss_raw[4] += metric_loss[1].item()
            

        # backward
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        writer.add_scalars(main_tag='Raw', tag_scalar_dict={'classify': loss_classify.item(), 
                                                            'center': loss_center.item(),
                                                            'homo': metric_loss[0].item(),
                                                            'heter': metric_loss[1].item()}, global_step=step)
        # vis gradient flow 
        if (1+i) % 100 == 0:
            writer.add_figure('Raw grad flow', plot_grad_flow_v2(net.named_parameters()), global_step=(i+1)//100)

        # Update Feature Center
        feature_center[y] += beta * (embeddings.detach() - feature_center[y])

        # metrics: top-1, top-3, top-5 error
        with torch.no_grad():
            epoch_acc[0] += accuracy(y_pred, y, topk=(1, 3, 5)).astype(np.float)

        if not options.raw:
            ##################################
            # Attention Cropping
            ##################################
            with torch.no_grad():
                crop_images = []
                for batch_index in range(attention_map.size(0)):
                    theta = torch.max(attention_map[batch_index]) * np.random.uniform(0.4, 0.6)
                    crop_mask = attention_map[batch_index] > theta
                    nonzero_indices = torch.nonzero(crop_mask[0, ...])
                    height_min = torch.clamp(nonzero_indices[:, 0].min().float() / attention_map.size(2) - 0.1, min=0) * X.size(2)
                    height_max = torch.clamp(nonzero_indices[:, 0].max().float() / attention_map.size(2) + 0.1, max=1) * X.size(2)
                    width_min = torch.clamp(nonzero_indices[:, 1].min().float() / attention_map.size(3) - 0.1, min=0) * X.size(3)
                    width_max = torch.clamp(nonzero_indices[:, 1].max().float() / attention_map.size(3) + 0.1, max=1) * X.size(3)
                    crop_images.append(F.upsample_bilinear(X[batch_index:batch_index + 1, :, int(height_min.item()):int(height_max.item()), 
                                                                int(width_min.item()):int(width_max.item())], size=crop_size))
                crop_images = torch.cat(crop_images, dim=0)
                #print('raw image:', X[0, 0, ...])
                #print('crop_images:', crop_images.size(), '\n', crop_images[0, 0, ...])

            # crop images forward
            y_pred, embeddings_cropped, _ = net(crop_images)

            # loss
            metric_loss = loss_metric(embeddings_cropped)
            loss_classify = loss(y_pred, y)
            #embeddings = embeddings.detach()
            #metric_l2 = l2_loss(embeddings_cropped, embeddings)
            #batch_loss = metric_loss[0] + metric_loss[1] + metric_l2 
            #if not options.freeze:
            batch_loss = loss_classify + metric_loss[0] + metric_loss[1]
            epoch_loss_crop[0] += batch_loss.item()
            epoch_loss_crop[1] += loss_classify.item()
            epoch_loss_crop[2] += metric_loss[0].item()
            epoch_loss_crop[3] += metric_loss[1].item()
            #epoch_loss[7] += metric_l2.item()


            # backward
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            if (1+i) % 100 == 0:
                writer.add_figure('Cropped grad flow', plot_grad_flow_v2(net.named_parameters()), global_step=(i+1)//100)

            # metrics: top-1, top-3, top-5 error
            with torch.no_grad():
                epoch_acc[1] += accuracy(y_pred, y, topk=(1, 3, 5)).astype(np.float)

            ##################################
            # Attention Dropping
            ##################################
            if not options.freeze:
                with torch.no_grad():
                    drop_mask = F.upsample_bilinear(attention_map, size=(X.size(2), X.size(3)))
                    for batch_index in range(attention_map.size(0)):
                        theta = torch.max(attention_map[batch_index]) * np.random.uniform(0.4, 0.6)
                        drop_mask[batch_index] = drop_mask[batch_index] < theta
                    drop_images = X * drop_mask.float()

                # drop images forward
                y_pred, _, _ = net(drop_images)

                # loss
                batch_loss = loss(y_pred, y)
                epoch_loss_drop[0] += batch_loss.item()

                # backward
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                # metrics: top-1, top-3, top-5 error
                with torch.no_grad():
                    epoch_acc[2] += accuracy(y_pred, y, topk=(1, 3, 5)).astype(np.float)

        # end of this batch
        batches += 1
        step += 1
        batch_end = time.time()
        if (i + 1) % verbose == 0:
            logging.info('\tBatch %d: (Raw) Loss %.3f (%.3f, %.3f, %.3f, %.3f), Accuracy: (%.2f, %.2f, %.2f), (Crop) Loss %.3f (%.3f, %.3f, %.3f), Accuracy: (%.2f, %.2f, %.2f), (Drop) Loss %.3f, Accuracy: (%.2f, %.2f, %.2f), Time %3.2f' %
                         (i + 1,
                          epoch_loss_raw[0] / batches, epoch_loss_raw[1] / batches, epoch_loss_raw[2] / batches, epoch_loss_raw[3] / batches, epoch_loss_raw[4] / batches,
                          epoch_acc[0, 0] / batches, epoch_acc[0, 1] / batches, epoch_acc[0, 2] / batches,
                          epoch_loss_crop[0] / batches, epoch_loss_crop[1] / batches, epoch_loss_crop[2] / batches, epoch_loss_crop[3] / batches, 
                          epoch_acc[1, 0] / batches, epoch_acc[1, 1] / batches, epoch_acc[1, 2] / batches,
                          epoch_loss_drop[0] / batches, epoch_acc[2, 0] / batches, epoch_acc[2, 1] / batches, epoch_acc[2, 2] / batches,
                          batch_end - batch_start))

    # save checkpoint model
    if epoch % save_freq == 0:
        state_dict = net.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict,
            'feature_center': feature_center.cpu()},
            os.path.join(save_dir, '%03d.ckpt' % (epoch + 1)))

    # end of this epoch
    end_time = time.time()

    # metrics for average
    epoch_loss_raw /= batches
    epoch_loss_crop /= batches
    epoch_loss_drop /= batches
    epoch_acc /= batches

    # show information for this epoch
    logging.info('Train: (Raw) Loss %.4f, Accuracy: (%.2f, %.2f, %.2f), (Crop) Loss %.4f, Accuracy: (%.2f, %.2f, %.2f), (Drop) Loss %.4f, Accuracy: (%.2f, %.2f, %.2f), Time %3.2f'%
                 (epoch_loss_raw[0], epoch_acc[0, 0], epoch_acc[0, 1], epoch_acc[0, 2],
                  epoch_loss_crop[0], epoch_acc[1, 0], epoch_acc[1, 1], epoch_acc[1, 2],
                  epoch_loss_drop[0], epoch_acc[2, 0], epoch_acc[2, 1], epoch_acc[2, 2],
                  end_time - start_time))


def validate(**kwargs):
    # Retrieve training configuration
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    loss = kwargs['loss']
    verbose = kwargs['verbose']
    #purpose = kwargs['purpose']

    # Default Parameters
    theta_c = 0.5
    crop_size = (256, 256)  # size of cropped images for 'See Better'

    # metrics initialization
    batches = 0
    epoch_loss = 0
    epoch_acc = np.array([0, 0, 0], dtype='float') # top - 1, 3, 5

    # begin validation
    start_time = time.time()
    net.eval()
    with torch.no_grad():
        for i, (X, y, _) in enumerate(data_loader):
            batch_start = time.time()

            # obtain data
            X = X.to(torch.device("cuda"))
            y = y.to(torch.device("cuda"))

            ##################################
            # Raw Image
            ##################################
            y_pred_raw, feature_matrix, attention_map = net(X)

            ##################################
            # Object Localization and Refinement
            ##################################
            '''
            crop_mask = F.upsample_bilinear(attention_map, size=(X.size(2), X.size(3))) > theta_c
            crop_images = []
            for batch_index in range(crop_mask.size(0)):
                nonzero_indices = torch.nonzero(crop_mask[batch_index, 0, ...])
                height_min = nonzero_indices[:, 0].min()
                height_max = nonzero_indices[:, 0].max()
                width_min = nonzero_indices[:, 1].min()
                width_max = nonzero_indices[:, 1].max()
                crop_images.append(F.upsample_bilinear(X[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max], size=crop_size))
            crop_images = torch.cat(crop_images, dim=0)

            y_pred_crop, _, _ = net(crop_images)
            '''

            # final prediction
            #y_pred = (y_pred_raw + y_pred_crop) / 2
            y_pred = y_pred_raw

            # loss
            batch_loss = loss(y_pred, y)
            epoch_loss += batch_loss.item()

            # metrics: top-1, top-3, top-5 error
            epoch_acc += accuracy(y_pred, y, topk=(1, 3, 5)).astype(np.float)

            # end of this batch
            batches += 1
            batch_end = time.time()
            if (i + 1) % verbose == 0:
                logging.info('\tBatch %d: Loss %.5f, Accuracy: Top-1 %.2f, Top-3 %.2f, Top-5 %.2f, Time %3.2f' %
                         (i + 1, epoch_loss / batches, epoch_acc[0] / batches, epoch_acc[1] / batches, epoch_acc[2] / batches, batch_end - batch_start))


    # end of validation
    end_time = time.time()

    # metrics for average
    epoch_loss /= batches
    epoch_acc /= batches

    # show information for this epoch
    logging.info('Valid: Loss %.5f,  Accuracy: Top-1 %.2f, Top-3 %.2f, Top-5 %.2f, Time %3.2f'%
                 (epoch_loss, epoch_acc[0], epoch_acc[1], epoch_acc[2], end_time - start_time))

    return epoch_loss


if __name__ == '__main__':
    main()
