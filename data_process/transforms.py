import numpy as np
import random
import torch
import torchvision
from torchvision.transforms import functional as F
import PIL

class SegmentationPresetTrain:
    def __init__(self, crop_size):
        min_size = int(0.95 * crop_size)
        max_size = int(1.20 * crop_size)
        trans = [
            RandomResize(min_size, max_size),
            RandomCrop(crop_size),
            # RandomHorizontalFlip(0.5),
            # RandomVerticalFlip(0.5),
            # Rotation(-20, 20),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        self.transforms = Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)

class SegmentationPresetEval:
    def __init__(self, crop_size):
        trans = [
            Resize(crop_size),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        self.transforms = Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)

def get_transform(train, crop_size = 224):
    if train:
        return SegmentationPresetTrain(crop_size)
    else:
        return SegmentationPresetEval(crop_size)

def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.resize(image, [self.size, self.size])
        target = F.resize(target, [self.size, self.size], interpolation=PIL.Image.NEAREST)
        return image, target

class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=PIL.Image.NEAREST)
        return image, target

class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target

class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=0) # If the image is too small, pad it with 0
        crop_params = torchvision.transforms.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target

class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class Rotation(object):
    def __init__(self, min_angle, max_angle):
        self.angle = random.randint(min_angle, max_angle)

    def __call__(self, image, target):
        image = F.rotate(image, self.angle)
        target = F.rotate(target, self.angle)
        return image, target