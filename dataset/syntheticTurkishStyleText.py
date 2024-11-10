from PIL import Image
import lmdb
import six

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.augmentations import CVGeometry, CVDeterioration, CVColorJitter
from dataset.charMapper import CharMapper


class SyntheticTurkishStyleText(Dataset):
    def __init__(self, args: dict):
        self.root = args["root"]
        self.args = args
        self.transforms = self._getTransforms()
        self.mapper = CharMapper(letters=args["letters"], maxLength=args["maxLength"])

        self.env = lmdb.open(self.root, readonly=True, lock=False, readahead=False, meminit=False)

        with self.env.begin(write=False) as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def __getitem__(self, index):
        key = self.keys[index]
        with self.env.begin(write=False) as txn:
            label = key.decode().split("-*-")[0]
            imgbuf = txn.get(key)
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            image = Image.open(buf).convert("RGB")
            image = self.transforms(image)
        length = torch.tensor(len(label) + 1)
        label = self.mapper(label)
        return image, label, length

    def __len__(self):
        return len(self.keys)

    @staticmethod
    def _getTransforms():
        return transforms.Compose([
            CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
            CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
            CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25),
            transforms.Resize((32, 128), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])
