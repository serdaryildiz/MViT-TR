import os

import torch
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
from dataset.charMapper import CharMapper


class STRIT(Dataset):
    def __init__(self, args: dict):
        self.root = args["root"]
        self.imgPaths = [os.path.join(self.root, p) for p in os.listdir(self.root)]
        self.transforms = self._getTransforms()

        self.mapper = CharMapper(letters=args["letters"], maxLength=args["maxLength"])

    def __getitem__(self, item):
        imgPath = self.imgPaths[item]
        image = Image.open(imgPath)
        image = self.transforms(image)
        label = imgPath.split('_')[-1].split('.')[0]
        length = torch.tensor(len(label) + 1)
        label = self.mapper(label)
        return image, label, length

    def __len__(self):
        return len(self.imgPaths)

    @staticmethod
    def _getTransforms():
        return transforms.Compose([
            transforms.Resize((32, 128), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])