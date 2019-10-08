"""
WS-DAN models

Hu et al.,
"See Better Before Looking Closer: Weakly Supervised Data Augmentation Network for Fine-Grained Visual Classification",
arXiv:1901.09891

Created: May 04,2019 - Yuchong Gu
Revised: May 07,2019 - Yuchong Gu
"""
import logging
import numpy as np
import torch
import torch.nn as nn

from models.vgg import VGG
from models.resnet import ResNet
from models.inception import *

__all__ = ['WSDAN']


# Bilinear Attention Pooling
class BAP(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B = features.size(0)
        M = attentions.size(1)

        for i in range(M):
            AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, 1, -1)
            if i == 0:
                feature_matrix = AiF
            else:
                feature_matrix = torch.cat([feature_matrix, AiF], dim=1)

        return feature_matrix


class BAP_v2(nn.Module):
    def forward(self, features, attentions):
        B, F, H, W = features.size()
        M = attentions.size(1)

        #features = features.view(B, F, -1).transpose(1, 2)
        #attentions = attentions.view(B, M, -1)
        #print(features.size(), attentions.size())

        I = torch.einsum('bmhw, bfhw -> bmf', attentions, features) # (B, M, F)
        #print(I.size())
        I /= H*W
        I = torch.mul(torch.sign(I), torch.sqrt(torch.abs(I)+1e-12))
        I = I.view(B, -1) # (B, M*F)
        
        return I.view(B, M, F)


# WS-DAN: Weakly Supervised Data Augmentation Network for FGVC
class WSDAN(nn.Module):
    def __init__(self, num_classes, M=32, metric_dim=512, net=None):
        super(WSDAN, self).__init__()
        self.num_classes = num_classes
        self.M = M
        self.dim = metric_dim

        # Default Network
        self.baseline = 'inception'
        self.num_features = 768
        self.expansion = 1

        # Network Initialization
        if net is not None:
            self.features = net.get_features()

            if isinstance(net, ResNet):
                self.baseline = 'resnet'
                self.expansion = self.features[-1][-1].expansion
                self.num_features = 512
            elif isinstance(net, VGG):
                self.baseline = 'vgg'
                self.num_features = 512
        else:
            self.features = inception_v3(pretrained=True).get_features()

        # Attention Maps
        '''
        att_conv = nn.Conv2d(self.num_features * self.expansion, self.M, kernel_size=1)
        att_conv.bias.data.fill_(0.)
        self.attentions = nn.Sequential(att_conv, nn.BatchNorm2d(self.M), nn.ReLU(inplace=True))
        '''
        self.attentions = nn.Sequential(nn.Conv2d(self.num_features * self.expansion, self.M, kernel_size=1, bias=False), nn.ReLU(inplace=True))

        # Bilinear Attention Pooling
        self.bap = BAP_v2()
        
        # Conv1d squeeze all attentions
        self.squeeze = nn.Conv2d(self.M, 1, 1)

        # Classification Layer
        self.fc = nn.Sequential(nn.BatchNorm1d(self.num_features * self.expansion), 
                                nn.Linear(self.num_features * self.expansion, self.num_classes))

        logging.info('WSDAN: using %s as feature extractor' % self.baseline)

    def forward(self, x):
        batch_size = x.size(0)

        # Feature Maps, Attention Maps and Feature Matrix
        feature_maps = self.features(x)
        #print('feature_maps:', feature_maps.size(), 'std:', feature_maps.view(batch_size, 768, -1).std(dim=2).mean())
        attention_maps = self.attentions(feature_maps)
        #print('attention_maps:', attention_maps.size(), 'zero-rate:', (attention_maps==0).sum().float() / attention_maps.numel(), '\tstd:', attention_maps.view(batch_size, self.M, -1).std(dim=2).mean())
        embeddings = self.bap(feature_maps, attention_maps) # (B, M, F)
        #print('BAP zero-rate:', (embeddings==0).sum().float() / embeddings.numel(), '\n', embeddings[0, ...])
        embeddings = self.squeeze(embeddings.unsqueeze(-1)).view(-1, self.num_features) # (B, F)
        embeddings = nn.functional.normalize(embeddings)
        #print('Embedding zero-rate:', (embeddings==0).sum().float() / embeddings.numel(), '\n', embeddings[0, :])

        # Classification
        p = self.fc(embeddings) # weird that original implementation in tensorflow multiplies a constant 100

        # Generate Attention Map
        #print('attention_maps:', attention_maps.size())
        H, W = attention_maps.size(2), attention_maps.size(3)
        if self.training:
            # Randomly choose one of attention maps Ak
            part_weights = attention_maps.mean(dim=(2, 3)) # (B, atts)
            #print(part_weights.size())
            part_weights = torch.sqrt(part_weights + 1e-12)
            part_weights = torch.div(part_weights, part_weights.sum(dim=1, keepdim=True))
            part_weights = part_weights.cpu().detach().numpy()
            attention_map = torch.zeros(batch_size, 1, H, W).to(torch.device("cuda"))  # (B, 1, H, W)
            for i in range(batch_size):
                #print('prob[%d]:'%i, part_weights[i, :])
                indice = np.random.choice(range(self.M), p=part_weights[i, :])
                attention_map[i] = attention_maps[i, indice:indice + 1, ...]
        else:
            # Object Localization Am = mean(sum(Ak))
            attention_map = torch.mean(attention_maps, dim=1, keepdim=True)  # (B, 1, H, W)
            #print('Test phase attention zero-rate', (attention_map==0).sum().float() / attention_map.numel())
            #print(attention_map[0, ...])
        '''
        # Normalize Attention Map
        attention_map = attention_map.view(batch_size, -1)  # (B, H * W)
        attention_map_max, _ = attention_map.max(dim=1, keepdim=True)  # (B, 1)
        attention_map_min, _ = attention_map.min(dim=1, keepdim=True)  # (B, 1)
        attention_map = (attention_map - attention_map_min) / (attention_map_max - attention_map_min)  # (B, H * W)
        attention_map = attention_map.view(batch_size, 1, H, W)  # (B, 1, H, W)
        '''

        return p, embeddings, attention_map

    def load_state_dict(self, state_dict, strict=True):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and model_dict[k].size() == v.size()}

        if len(pretrained_dict) == len(state_dict):
            logging.info('%s: All params loaded' % type(self).__name__)
        else:
            logging.info('%s: Some params were not loaded:' % type(self).__name__)
            not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
            logging.info(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))

        model_dict.update(pretrained_dict)
        super(WSDAN, self).load_state_dict(model_dict)


if __name__ == '__main__':
    net = WSDAN(num_classes=1000)
    '''
    net.train()

    for i in range(10):
        input_test = torch.randn(10, 3, 512, 512)
        p, feature_matrix, attention_map = net(input_test)

    print(p.shape)
    print(feature_matrix.shape)
    print(attention_map.shape)
    '''
    params = []
    params.extend(list(net.compact.parameters()))
    params.extend(list(net.metric.parameters()))
    print(params)
    '''
    net.eval()
    input_test = torch.randn(2, 3, 512, 352)
    with torch.no_grad():
        p, feature_matrix, attention_map = net(input_test)

    print(p.shape)
    print(feature_matrix.shape)
    print(attention_map.shape)
    '''
