import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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
                for j in range(i+1, self.batch_k):
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
            n = n.replace('.weight', '').replace('inception', '').replace('branch', 'B')
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    #print('layers:', layers)
    fig = plt.figure(1, figsize=(20, 5))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
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
