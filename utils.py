import numpy as np
import torch


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
        return d.sum()
    else:
        return torch.clamp(1-d, min=0).sum()


class MetricLoss(torch.nn.Module):
    def __init__(self, batch_k):
        super().__init__()
        self.batch_k = batch_k

    def forward(self, x):
        assert x.size(0) % self.batch_k == 0
        loss_homo, loss_heter = 0, 0
        batch_size = x.size(0)
        for group_index in range(batch_size//self.batch_k):
            for i in range(self.batch_k):
                anchor = x[i+group_index*self.batch_k:1+i+group_index*self.batch_k, ...]
                # loss from same label
                for j in range(i+1, self.batch_k):
                    loss_homo += L_metric(anchor, x[j+group_index*self.batch_k: 1+j+group_index*self.batch_k, ...])
                # loss from different label
                for j in range((1+group_index)*self.batch_k, batch_size):
                    loss_heter += L_metric(anchor, x[j: j+1, ...], same_class=False)
        return 2*loss_homo / (batch_size*(self.batch_k-1)), 2*loss_heter / (batch_size*(batch_size - self.batch_k))
