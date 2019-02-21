# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch,cv2
import torchvision

import os
from PIL import Image
from torch.utils.data import Dataset

class customDataSet(Dataset):

    @staticmethod
    def get_images(path):
        result = []
        for id, file in enumerate(os.listdir(path)):
            image_path = path + "/" + file
            # image = cv2.imread(image_path)
            image = Image.open(image_path).convert("RGB")
            image_info = {"id":id,
                          "image": image,
                          "width": image.size[0],
                          "height": image.size[1],
                          "path": image_path
                          }
            result.append(image_info)
        return result

    def __init__(self,data_path,transfrom):
        self.random_target = torch.randn(3,3)
        self.transfrom = transfrom
        self.image_list = customDataSet.get_images(data_path)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = self.image_list[idx].get("image")
        # image, random_target = self.transfrom(image,self.random_target)
        image = self.transfrom(image)
        return image, idx

    def get_img_info(self, index):
        img_data = self.image_list[index]
        return img_data