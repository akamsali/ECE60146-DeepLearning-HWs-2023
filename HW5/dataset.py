import torch
from torchvision import transforms as tvt

from PIL import Image
import pickle

import os

from typing import Union


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, categories: list, split="train") -> None:
        super().__init__()

        self.split = split
        self.transform = tvt.Compose(
            [
                tvt.PILToTensor(),
                tvt.ConvertImageDtype(torch.float),
                tvt.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.set_classes(categories=categories)

    def set_classes(self, categories: list) -> None:
        self.categories = categories
        # self.label_dict = {category: i for i, category in enumerate(self.categories)}
        self.file_list = []
        self.label_list = []
        self.bbox_list = []
        for label, category in enumerate(self.categories):
            split_path = f'./{self.split}_manifest_{category}.pkl'
            with open(split_path, 'rb') as handle:
                data = pickle.load(handle)
            
            split_list = list(
                map(lambda x: x['file_name'], data)
            )
            split_bboxes = list(map(lambda x: x['bboxes'][0], data))
            split_lables = [label] * len(split_list)

            self.file_list += split_list
            self.label_list += split_lables
            self.bbox_list += split_bboxes

    def __len__(self) -> int:
        if len(self.file_list) == len(self.label_list):
            return len(self.file_list)

    def __getitem__(self, index: int) -> Union[Image.Image, int]:
        # read image with PIL.Image.open
        img = Image.open(self.file_list[index])
        # transform image with self.transform
        img = self.transform(img)
        label = self.label_list[index]
        bbox = self.bbox_list[index]

        return img, label, bbox
