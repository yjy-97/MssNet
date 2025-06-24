from PIL import Image
import torch
from torch.utils.data import Dataset
import logging
import torch
from torch.utils import data
from torchvision import transforms, datasets
import os
from PIL import Image
import numpy as np
import albumentations as alb
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

class MyDataSet(Dataset):
    def __init__(self, root):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, k) for k in imgs]
        #np.random.shuffle(self.imgs)
        self.transforms = transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = img_path[-27]
        if label == 'N':
            label = 1
        else:
            label = 0

        pil_img = Image.open(img_path).convert('RGB')
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img)
            data = torch.from_numpy(pil_img)
        return data, label

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
