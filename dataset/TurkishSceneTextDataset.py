import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.augmentations import CVGeometry, CVDeterioration, CVColorJitter
from dataset.charMapper import CharMapper


class TurkishSceneTextDataset(Dataset):
    def __init__(self, args: dict, train=True):
        self.root = args["root"]
        self.train = train

        txt_path = os.path.join(self.root, "train.txt") if self.train else os.path.join(self.root, "test.txt")
        with open(txt_path, "r") as fp:
            lines = fp.readlines()

        self.gt = {}
        for l in lines:
            img_name, label = l.strip().split('\t')
            self.gt[img_name] = label

        self.root = os.path.join(self.root, "train") if self.train else os.path.join(self.root, "test")
        self.imgPaths = [os.path.join(self.root, p) for p in self.gt.keys()]

        self.transforms = self._getTransforms()
        self.mapper = CharMapper(letters=args["letters"], maxLength=args["maxLength"])

    def __getitem__(self, item):
        imgPath = self.imgPaths[item]
        image = Image.open(imgPath)
        image = self.transforms(image)

        label = self.gt[os.path.basename(imgPath)]

        label, length = self.mapper(label, return_length=True)
        return image, label, torch.tensor(length)

    def __len__(self):
        return len(self.imgPaths)

    def _getTransforms(self):
        if self.train:
            return transforms.Compose([
                CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
                CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
                CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25),
                transforms.Resize((32, 128), transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)
            ])
        else:
            return transforms.Compose([
                transforms.Resize((32, 128), transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)
            ])


if __name__ == '__main__':
    args = {
        "name": "TurkishSceneTextDataset",
        "root": "../data/TS-TR",
        "letters": "0123456789abcçdefgğhıijklmnoöpqrsştuüvwxyz",
        "maxLength": "25"
    }
    dataset = TurkishSceneTextDataset(args, False)
