import torch
from torchvision import transforms as tvt

from PIL import Image
import os

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root: "str", categories: list, split='train') -> None:
        super().__init__()
        # set root file
        self.root = root
        self.split = split
        self.transform = tvt.Compose(
            [
                tvt.PILToTensor(),
                tvt.ConvertImageDtype(torch.float),
                tvt.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                
            ]
        )
        self.set_classes(categories=categories)

    def set_classes(self, categories):
        self.categories = categories
        # self.label_dict = {category: i for i, category in enumerate(self.categories)}
        self.file_list = []
        self.label_list = []
        for label, category in enumerate(self.categories):
            split_path = os.path.join(self.root, category, self.split)
            split_list = list(map(lambda x: os.path.join(split_path, x), os.listdir(split_path)))
            split_lables = [label]*len(split_list)

            self.file_list += split_list
            self.label_list += split_lables

    def __len__(self) -> int:
        if len(self.file_list) == len(self.label_list):
            return len(self.file_list)

    def __getitem__(self, index: int):
        # read image with PIL.Image.open
        img = Image.open(self.file_list[index])
        # transform image with self.transform
        img = self.transform(img)
        label = self.label_list[index]

        return img, label