""" NOTICE: A Custom Dataset SHOULD BE PROVIDED
Created: May 02,2019 - Yuchong Gu
Revised: May 07,2019 - Yuchong Gu
"""
import os
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import pickle
import numpy as np

__all__ = ['CustomDataset', 'ImageFolderWithName', 'CustomSampler']


config = {
    # e.g. train/val/test set should be located in os.path.join(config['datapath'], 'train/val/test')
    'datapath': '/home/chk/car_kaggle/',
}


class CustomDataset(Dataset):
    """
    # Description:
        Basic class for retrieving images and labels

    # Member Functions:
        __init__(self, phase, shape):   initializes a dataset
            phase:                      a string in ['train', 'val', 'test']
            shape:                      output shape/size of an image

        __getitem__(self, item):        returns an image
            item:                       the idex of image in the whole dataset

        __len__(self):                  returns the length of dataset
    """

    def __init__(self, phase='train', shape=512):
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.data_path = os.path.join(config['datapath'], phase)
        self.data_list = os.listdir(self.data_path)

        self.shape = shape
        self.config = config

        # transform
        self.transform = transforms.Compose([
            transforms.Resize(size=self.shape),
            transforms.RandomCrop((self.shape, self.shape)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, item):
        image = Image.open(os.path.join(self.data_path, self.data_list[item])).convert('RGB')  # (C, H, W)
        image = self.transform(image)
        assert image.size(1) == self.shape and image.size(2) == self.shape

        if self.phase != 'test':
            # filename of image should have 'id_label.jpg/png' form
            label = int((self.data_list[item].split('.')[0]).split('_')[-1])  # label
            return image, label
        else:
            # filename of image should have 'id.jpg/png' form, and simply return filename in case of 'test'
            return image, self.data_list[item]

    def __len__(self):
        return len(self.data_list)


class ImageFolderWithName(datasets.ImageFolder):
    def __init__(self, phase='train', shape=512, *args, **kwargs):
        self.transforms = transforms.Compose([
            transforms.Resize(shape),
            transforms.RandomCrop((shape, shape)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        assert phase in ['train', 'val', 'test']
        root = os.path.join(config['datapath'], phase)
        super().__init__(root=root, transform=self.transforms, *args, **kwargs)
        self.return_fn = (phase == 'test') or (phase == 'val')

    def __getitem__(self, i):
        img, label = super(ImageFolderWithName, self).__getitem__(i)
        assert label <= 98*2
        if not self.return_fn:
            return img, label
        else:
            return img, name2id[label], self.imgs[i]


class CustomSampler(Sampler):
    def __init__(self, dataset, batch_size=32, batch_k=4, len=5000):
        assert batch_size % batch_k == 0
        self.batch_size = batch_size
        self.batch_k = batch_k
        self.len = len
        self.classes_per_batch = int(batch_size / batch_k)
        self.labels = np.array([data[1] for data in dataset.imgs])
        self.unique_labels = np.unique(self.labels)

    def __iter__(self):
        count = 0
        for i in range(len(self.labels)):
            class_ids = np.random.choice(self.unique_labels, self.classes_per_batch, replace=False)
            indices = []
            for label in class_ids:
                indices.extend(np.random.choice(np.nonzero(self.labels == label)[0], self.batch_k, replace=False))
            yield indices
            count += 1
            if count >= self.len:
                break


if __name__ == '__main__':
    pass
