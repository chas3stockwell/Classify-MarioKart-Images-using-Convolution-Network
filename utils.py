import csv
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F

from . import dense_transforms

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']
DENSE_LABEL_NAMES = ['background', 'kart', 'track', 'bomb/projectile', 'pickup/nitro']
# Distribution of classes on dense training set (background and track dominate (96%)
DENSE_CLASS_DISTRIBUTION = [0.52683655, 0.02929112, 0.4352989, 0.0044619, 0.00411153]


LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']
dict_label = {'background':0, 'kart':1, 'pickup':2, 'nitro':3, 'bomb':4, 'projectile':5}

transform_norm = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.32343397, 0.33100516, 0.34438375], std=[0.16127683, 0.13571456, 0.16258068]),
            transforms.RandomHorizontalFlip(p=0.5),
            #transforms.Pad(2)
            #transforms.ColorJitter(brightness=0) #havent found right ratio yet. This is what I want to mess with. Averages 86% without any augmentation right now. 
        ])

class SuperTuxDataset(Dataset):
    
    def __init__(self, dataset_path, transform=transforms.ToTensor()):
        
        with open(dataset_path+'/labels.csv', newline='\n') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            headers = next(spamreader, None)
            #print(headers)
            self.data = []
            for row in spamreader:
                self.data.append( {
                    list(zip(headers, row))[0][0]: dataset_path+"/"+list(zip(headers, row))[0][1],
                    list(zip(headers, row))[1][0]: dict_label[list(zip(headers, row))[1][1]],
                })
            #print(self.data)
        self.transform = transform
        
                 
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        im = Image.open(self.data[idx]['file'])
        if self.transform is not None:
            im = (self.transform(im))
        return im, self.data[idx]['label']

        
    
test = [ dense_transforms.RandomHorizontalFlip(), dense_transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5), dense_transforms.ToTensor(), ]
trans = dense_transforms.Compose([dense_transforms.RandomHorizontalFlip(), dense_transforms.ColorJitter(brightness=0.75, contrast=0.75, hue=0.25), dense_transforms.ToTensor() ])


class DenseSuperTuxDataset(Dataset):
    def __init__(self, dataset_path, transform=trans):
        from glob import glob
        from os import path
        print(dataset_path)
        self.files = []
        for im_f in glob(path.join(dataset_path, '*_im.jpg')):
            self.files.append(im_f.replace('_im.jpg', ''))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        b = self.files[idx]
        im = Image.open(b + '_im.jpg')
        lbl = Image.open(b + '_seg.png')
        if self.transform is not None:
            im, lbl = self.transform(im, lbl)
        return im, lbl


def load_data(dataset_path, num_workers=0, batch_size=128, **kwargs):
    dataset = SuperTuxDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def load_dense_data(dataset_path, num_workers=0, batch_size=32,  **kwargs):
    dataset = DenseSuperTuxDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def _one_hot(x, n):
    return (x.view(-1, 1) == torch.arange(n, dtype=x.dtype, device=x.device)).int()


class ConfusionMatrix(object):
    def _make(self, preds, labels):
        label_range = torch.arange(self.size, device=preds.device)[None, :]
        preds_one_hot, labels_one_hot = _one_hot(preds, self.size), _one_hot(labels, self.size)
        return (labels_one_hot[:, :, None] * preds_one_hot[:, None, :]).sum(dim=0).detach()

    def __init__(self, size=5):
        """
        This class builds and updates a confusion matrix.
        :param size: the number of classes to consider
        """
        self.matrix = torch.zeros(size, size)
        self.size = size

    def add(self, preds, labels):
        """
        Updates the confusion matrix using predictions `preds` (e.g. logit.argmax(1)) and ground truth `labels`
        """
        self.matrix = self.matrix.to(preds.device)
        self.matrix += self._make(preds, labels).float()

    @property
    def class_iou(self):
        true_pos = self.matrix.diagonal()
        #print(true_pos) #number of true positives 

        #print(self.matrix)
        #print(self.matrix.sum(0))
        #print(self.matrix.sum(1))
        return true_pos / (self.matrix.sum(0) + self.matrix.sum(1) - true_pos + 1e-5)

    @property
    def iou(self):
        return self.class_iou.mean()

    @property
    def global_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos.sum() / (self.matrix.sum() + 1e-5)

    @property
    def class_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(1) + 1e-5)

    @property
    def average_accuracy(self):
        return self.class_accuracy.mean()

    @property
    def per_class(self):
        return self.matrix / (self.matrix.sum(1, keepdims=True) + 1e-5)


if __name__ == '__main__':
    dataset = DenseSuperTuxDataset('dense_data/dense_data/train', transform=dense_transforms.Compose(
        [dense_transforms.RandomHorizontalFlip(), dense_transforms.ToTensor(), dense_transforms.ColorJitter(), dense_transforms.Normalize()]))
    from pylab import show, imshow, subplot, axis

    for i in range(15):
        im, lbl = dataset[i]
        subplot(5, 6, 2 * i + 1)
        imshow(F.to_pil_image(im))
        axis('off')
        subplot(5, 6, 2 * i + 2)
        imshow(dense_transforms.label_to_pil_image(lbl))
        axis('off')
    show()
    import numpy as np

    c = np.zeros(5)
    for im, lbl in dataset:
        c += np.bincount(lbl.view(-1), minlength=len(DENSE_LABEL_NAMES))
    print(100 * c / np.sum(c))
