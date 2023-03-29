import torch
from torchvision import transforms as tvt

import os
from PIL import Image
from typing import Union




class MyDataset(torch.utils.data.Dataset):
    def __init__(self, 
                data_path='/mnt/cloudNAS4/akshita/Documents/datasets/pizza', 
                split="train") -> None:
        super().__init__()

        self.split = split
        self.transform = tvt.Compose(
            [
                tvt.PILToTensor(),
                tvt.ConvertImageDtype(torch.float),
                tvt.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.get_file_list(data_path)

    def get_file_list(self, data_path) -> None:
        # self.file_list = []

        data_path = os.path.join(data_path, self.split)
        self.file_list = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith('.jpg')]
        

    def __len__(self) -> int:
        # if len(self.file_list) == len(self.label_list):
        return len(self.file_list)

    def __getitem__(self, index: int) -> Union[Image.Image, int]:
        # read image with PIL.Image.open
        img = Image.open(self.file_list[index])
        return self.transform(img)

