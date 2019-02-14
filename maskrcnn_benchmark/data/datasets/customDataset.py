# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision

import os
from PIL import Image
from torch.utils.data import Dataset

class customDataSet(Dataset):

    @staticmethod
    def get_image_list(path):
        result = []
        for file in os.listdir(path):
            result.append(path + "/" + file)
        return result

    def __init__(self,data_path,transfrom):
        self.transfrom = transfrom
        self.image_list = customDataSet.get_image_list(data_path)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx])
        image = self.transfrom(image)
        return image, idx
