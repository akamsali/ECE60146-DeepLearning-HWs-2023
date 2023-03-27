import torch
from torchvision import transforms as tvt

from PIL import Image
import pickle

import os

from typing import Union


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, categories: list, split="train",
                  manifest_path='./manifests', mac=False) -> None:
        super().__init__()

        self.split = split
        self.transform = tvt.Compose(
            [
                tvt.PILToTensor(),
                tvt.ConvertImageDtype(torch.float),
                tvt.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.set_classes(categories=categories, manifest_path=manifest_path)
        self.mac = mac

    def clip_file_names(self, a,
                        path='/Users/akshita/Documents/Acads/data/coco_custom_HW6'):
        if self.mac:
            img_path = a.split('/')[-3:]
            return(os.path.join(path, *img_path))
        else:
            return a

    def set_classes(self, categories: list, manifest_path: str) -> None:
        self.categories = categories
        # self.label_dict = {category: i for i, category in enumerate(self.categories)}
        self.file_list = []
        self.label_list = []
        self.bbox_list = []
        for label, category in enumerate(self.categories):
            split_path = f'{manifest_path}/{self.split}_manifest_{category}.pkl'
            with open(split_path, 'rb') as handle:
                data = pickle.load(handle)
            
            split_list = list(
                map(lambda x: x['file_name'], data)
            )
            # for x in data:
            #     print(x['bboxes'])
            split_bboxes = list(map(lambda x: x['bboxes'], data))
            # print(split_bboxes)
            # split_bboxes = 
            # print(bboxes)

            split_lables = [label] * len(split_list)

            self.file_list += split_list
            self.label_list += split_lables
            self.bbox_list += split_bboxes

    def __len__(self) -> int:
        if len(self.file_list) == len(self.label_list):
            return len(self.file_list)

    def get_yolo_vectors(self, bboxes, class_idx, yolo_interval=32, tot_anch_boxes=5, img_size=(256,256)):
        num_cells_image_width = int(img_size[0] / yolo_interval)
        num_cells_image_height = int(img_size[1] / yolo_interval)
        # print(num_cells_image_width, num_cells_image_height)
        anchor_box_indices, yolo_cell_indices, yolo_vectors = [], [], []
        yolo_tensor = torch.zeros((num_cells_image_width*num_cells_image_height, 
                                    tot_anch_boxes, 
                                    8
                                    ))
        for idx, bbox in enumerate(bboxes): 
            x, y, w, h = bbox

            if (w < 16.0) or (h < 16.0): 
                continue

            # get the centre of the bbox
            height_center_bb =  y + h//2
            width_center_bb =  w + w//2
            
            # get the row and column index of the centre of the bbox
            cell_row_indx =  int(height_center_bb / yolo_interval)          
            cell_col_indx =  int(width_center_bb / yolo_interval)
            # print(idx, cell_row_indx, cell_col_indx)
            cell_row_indx = min(max(0, cell_row_indx), num_cells_image_width-1)
            cell_col_indx = min(max(0, cell_col_indx), num_cells_image_height-1)

            b_w, b_h = w/yolo_interval, h/yolo_interval

            yolocell_center_i =  cell_row_indx*yolo_interval + float(yolo_interval) / 2.0
            yolocell_center_j =  cell_col_indx*yolo_interval + float(yolo_interval) / 2.0
            del_x  =  (width_center_bb - yolocell_center_j) / yolo_interval
            del_y  =  (height_center_bb - yolocell_center_i) / yolo_interval
            # print(del_x, del_y)
            AR = h / w
            # print(AR)
            anch_box_index = None
            if AR <= 0.2:               anch_box_index = 0                                                     ## (45)
            if 0.2 < AR <= 0.5:         anch_box_index = 1                                                     ## (46)
            if 0.5 < AR <= 1.5:         anch_box_index = 2                                                     ## (47)
            if 1.5 < AR <= 4.0:         anch_box_index = 3                                                     ## (48)
            if AR > 4.0:                anch_box_index = 4
            
            yolo_vector = torch.FloatTensor([1, del_x, del_y, b_w, b_h, 0, 0, 0])
            yolo_vector[5+class_idx] = 1
            yolo_cell_idx = cell_row_indx * num_cells_image_width  +  cell_col_indx
            
            yolo_tensor[yolo_cell_idx, anch_box_index, :] = yolo_vector

            anchor_box_indices.append(anch_box_index)
            yolo_cell_indices.append(yolo_cell_idx)
            yolo_vectors.append(yolo_vector)

        return anchor_box_indices, yolo_cell_indices, yolo_vectors, yolo_tensor

    def __getitem__(self, index: int) -> Union[Image.Image, int]:
        # read image with PIL.Image.open
        img = Image.open(self.clip_file_names(self.file_list[index]))
        # transform image with self.transform
        img = self.transform(img)
        label = self.label_list[index]
        bboxes = self.bbox_list[index]

        if self.split == "train":

            _, _, _, yolo_tensor = self.get_yolo_vectors(bboxes,class_idx=label)
            
            yolo_tensor_aug = torch.cat((yolo_tensor, 
                                        torch.zeros((*yolo_tensor.shape[:-1], 1))), dim=-1)

            # print(bbox)
            for i in range(yolo_tensor_aug.shape[0]):
                for j in range(yolo_tensor_aug.shape[1]):
                    if yolo_tensor_aug[i, j, 0] == 0:
                        yolo_tensor_aug[i, j, -1] == 1
            return img, label, yolo_tensor_aug
        elif self.split == "val":
            return img, label, bboxes
