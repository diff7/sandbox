import random
import copy
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

# from torchvision import transforms
import torchvision.transforms.functional as F

to_tensor = transforms.ToTensor()
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def double_transforms(image, num_cuts_off=10, mean=0.0, std=1):
    # image in PIL format
    im_weak = copy.copy(image)
    im_strong = copy.copy(image)

    # Random crop
    w_, h_ = image.size
    wcrop, hcrop = int(w_ * 0.8), int(h_ * 0.8)
    i, j, h, w = transforms.RandomCrop.get_params(im_weak, (wcrop, hcrop))
    im_weak, im_strong = F.crop(im_weak, i, j, h, w), F.crop(im_strong, i, j, h, w)
    # Random flip
    if random.random() > 0.5:
        im_weak, im_strong = F.hflip(im_weak), F.hflip(im_strong)

    # Random brightness
    factor = 1 - 0.5 * random.random()
    im_strong = F.adjust_brightness(img=im_strong, brightness_factor=factor)
    im_weak, im_strong = normalize(to_tensor(im_weak)), to_tensor(im_strong)

    # erase some areas
    size = w_
    for i in range(num_cuts_off):
        wr = int(size * random.random())
        hr = int(size * random.random())
        b = int(size * 0.1 * random.random())
        im_strong = F.erase(im_strong, wr, hr, b, b, v=int(random.random() > 0.5))

    # gaussian noise
    im_strong += torch.randn(im_strong.size()) * std + mean
    return im_weak, im_strong


class DictLoaderWithTransforms(Dataset):
    """ Input data: list of dicts path, label
        Returns: tensor, label """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        path = self.data[idx]["path"]
        true_label = self.data[idx]["label"]
        orig_img = Image.open(path)

        try:
            im_weak, im_strong = double_transforms(orig_img)
        except Exception as e:
            print(e)

        return im_weak, im_strong, true_label


class LoaderWithTransforms(Dataset):
    """ Input data: tensors
        Returns: tensor, label """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        orig_img = self.data[0][idx]
        orig_img = Image.fromarray(np.uint8(orig_img))
        true_label = self.data[1][idx]
        try:
            im_weak, im_strong = double_transforms(orig_img)
        except Exception as e:
            print(e)

        return im_weak, im_strong, true_label


def process_one_image(img_path):
    img = Image.open(img_path)
    img = to_tensor(img)
    img = normalize(img)
    return img


class DictLoaderSimple(Dataset):
    """ Input data: list of dicts, path, label
        Returns: tensor, label """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]["path"]
        true_label = self.data[idx]["label"]
        img = process_one_image(orig_path)

        return true_label, true_label
