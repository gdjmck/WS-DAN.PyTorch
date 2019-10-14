import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import re


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100. / batch_size))

    return np.array(res)


def L_metric(feat1, feat2, same_class=True):
    d = torch.sum((feat1 - feat2).pow(2).view((-1, feat1.size(-1))), 1)
    if same_class:
        return d.sum() / d.size(0)
    else:
        return torch.clamp(1-d, min=0).sum() / d.size(0)


def center_loss(features, centers):
    '''
        both features and centers are of shape (N, dim)
    '''
    #print('features:', features.size(), '\ncenters:', centers.size())
    batch_size = features.size(0)
    centers = torch.nn.functional.normalize(centers)
    distance = (features - centers) ** 2
    return distance.sum() / batch_size


class MetricLoss(torch.nn.Module):
    def __init__(self, batch_k):
        super().__init__()
        self.batch_k = batch_k

    def forward(self, x):
        assert x.size(0) % self.batch_k == 0
        loss_homo, loss_heter = 0, 0
        cnt_homo, cnt_heter = 0, 0
        batch_size = x.size(0)
        for group_index in range(batch_size//self.batch_k):
            for i in range(self.batch_k):
                anchor = x[i+group_index*self.batch_k:1+i+group_index*self.batch_k, ...]
                # loss from same label
                for j in range(self.batch_k):
                    if i == j: continue
                    loss_homo += L_metric(anchor, x[j+group_index*self.batch_k: 1+j+group_index*self.batch_k, ...])
                    cnt_homo += 1
                # loss from different label
                for j in range((1+group_index)*self.batch_k, batch_size):
                    loss_heter += L_metric(anchor, x[j: j+1, ...], same_class=False)
                    cnt_heter += 1
        return loss_homo / cnt_homo, loss_heter / cnt_heter


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    fig = plt.figure(1)
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    return fig
    

def plot_grad_flow_v2(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            if p.grad is None:
                #print('no grad:', n)
                continue
            n = n.replace('.weight', '').replace('inception', '').replace('branch', 'B').replace('conv', 'C').replace('pool', 'P').replace('attentions', 'A').replace('_', '.').replace('features.', '')
            #n = re.sub('\d', '', n)
            #n.replace('..', '.')
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    #print('layers:', layers)
    fig = plt.figure(1, figsize=(20, 5))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical", fontsize='xx-small')
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    return fig

def rescale_padding(tensor, size):
    scale = float(size) / max(tensor.size(-1), tensor.size(-2))
    target = torch.nn.functional.interpolate(tensor, scale_factor=scale)
    if target.size(2) >= target.size(3):
        # h > w, padding to left and right
        margin = size - target.size(3)
        pad = [margin//2, margin-margin//2, size-target.size(2), 0]
    else:
        margin = size - target.size(2)
        pad = [size-target.size(3), 0, margin//2, margin-margin//2]
    target = torch.nn.functional.pad(target, pad)
    return target

def rescale_central(tensor, borders, target_size):
    H, W = tensor.size(-2), tensor.size(-1)
    top, bottom, left, right = borders
    h, w = bottom - top, right - left
    gap = abs(h-w)
    if h > w:
        left -= gap//2
        right = left + h
        if left < 0:
            right -= left
            left = 0
        if right >= W:
            left -= right - W + 1
            right = W - 1
    elif w > h:
        top -= gap//2
        bottom = top + w
        if top < 0:
            bottom -= top
            top = 0
        if bottom >= H:
            top -= bottom - H + 1
            bottom = H - 1
    assert (bottom - top) == (right - left)
    
    return torch.nn.functional.interpolate(tensor[..., top: bottom, left: right], (target_size, target_size))