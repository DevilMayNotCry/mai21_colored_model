# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

from torch.utils.data import Dataset
from torchvision import transforms
from scipy import misc
import numpy as np
import imageio
import torch
import os
import cv2
to_tensor = transforms.Compose([
    transforms.ToTensor()
])


def extract_bayer_channels(raw):

    # Reshape the input bayer image

    ch_B  = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R  = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]

    RAW_combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
    RAW_norm = RAW_combined.astype(np.float32) / (4 * 255)

    return RAW_norm


class LoadData(Dataset):

    def __init__(self, dataset_dir, dataset_size, dslr_scale, test=False):

        if test:
            self.raw_dir = os.path.join(dataset_dir, 'test', 'raw')
            self.dslr_dir = os.path.join(dataset_dir, 'test', 'retouched')
            self.dataset_size = dataset_size
        else:
            self.raw_dir = os.path.join(dataset_dir, 'train', 'raw')
            self.dslr_dir = os.path.join(dataset_dir, 'train', 'retouched')
        self.image_list = os.listdir(self.raw_dir)
        self.dataset_size = dataset_size
        self.scale = dslr_scale
        self.test = test

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        raw_image = misc.imread(os.path.join(self.raw_dir, self.image_list[idx]))
        raw_image = np.asarray(raw_image)
        raw_image = cv2.resize(raw_image, (raw_image.shape[0]//2, raw_image.shape[1]//2))
        #raw_image = extract_bayer_channels(raw_image)
        #raw_image = torch.from_numpy(raw_image)
        #raw_image = np.float32(np.reshape(raw_image, [1, raw_image.shape[0], raw_image.shape[1], raw_image.shape[2]]))

        dslr_image = misc.imread(os.path.join(self.dslr_dir, self.image_list[idx]))
        dslr_image = np.asarray(dslr_image)
        dslr_image = np.float32(misc.imresize(dslr_image, self.scale / 2)) / 255.0
        #dslr_image = torch.from_numpy(dslr_image.transpose((2, 0, 1)))

        return raw_image, dslr_image


class LoadVisualData(Dataset):

    def __init__(self, data_dir, size, scale, level, full_resolution=False):

        self.raw_dir = os.path.join(data_dir, 'test', 'raw')

        self.dataset_size = size
        self.scale = scale
        self.level = level
        self.full_resolution = full_resolution
        self.test_images = os.listdir(self.raw_dir)

        if level > 1 or full_resolution:
            self.image_height = 1440
            self.image_width = 1984
        elif level > 0:
            self.image_height = 1280
            self.image_width = 1280
        else:
            self.image_height = 960
            self.image_width = 960

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):

        raw_image = np.asarray(imageio.imread(os.path.join(self.raw_dir, self.test_images[idx])))
        raw_image = extract_bayer_channels(raw_image)

        if self.level > 1 or self.full_resolution:
            raw_image = raw_image[0:self.image_height, 0:self.image_width, :]
        elif self.level > 0:
            raw_image = raw_image[80:self.image_height + 80, 352:self.image_width + 352, :]
        else:
            raw_image = raw_image[240:self.image_height + 240, 512:self.image_width + 512, :]

        raw_image = torch.from_numpy(raw_image.transpose((2, 0, 1)))

        return raw_image
