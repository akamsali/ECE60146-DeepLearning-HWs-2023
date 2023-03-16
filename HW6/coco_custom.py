from pycocotools.coco import COCO

import cv2
import os
import pickle 

# import numpy as np
from tqdm import tqdm
# from typing import Union


class COCO_loader:
    def __init__(
        self,
        dataDir="/mnt/cloudNAS4/jo/coco",
        dataType="train",
    ) -> None:
        
        self.data_dir = dataDir
        self.data_type = dataType

        self.annFile = "{}/annotations/instances_{}2014.json".format(
            self.data_dir, self.data_type
        )
        # initialize COCO api for instance annotations
        self.coco = COCO(self.annFile)
        # self.cats = self.coco.loadCats(self.coco.getCatIds())
    
    # def set_inverse_labels(self, categories_list: list) -> None:
    #     self.catIds = self.coco.getCatIds(catNms=categories_list)
    #     self.categories = self.coco.loadCats(self.catIds)
    #     self.categories.sort(key=lambda x: x['id'])
    #     self.categories_labels_inverse = {}
    #     for idx, in_class in enumerate(categories_list):
    #         for c in self.categories:
    #             if c['name'] == in_class:
    #                 self.categories_labels_inverse[c['id']] = idx

        # self.imgIds = {self.categories_labels_inverse[i][1]: coco.getImgIds(catIds=[i]) 
        #                for i in catIds}
        
    
    def save_images_to_folder(
        self,
        category: str,
        data_path="/mnt/cloudNAS4/akshita/Documents/datasets/coco_custom_HW6",
        # dataType = 'train',
        size = (256, 256),
        min_area = 40000, 
        manifest_path = None,
    ) -> None:

        path = os.path.join(data_path, category)

        if not os.path.exists(os.path.join(path, self.data_type)):
            os.makedirs(os.path.join(path, self.data_type))
        
        catIds = self.coco.getCatIds(catNms=category)
        imgIds = self.coco.getImgIds(catIds=catIds)
        img_dict = self.coco.loadImgs(ids=imgIds)


        coco_path = os.path.join(self.data_dir, self.data_type)
        final_path = os.path.join(path, self.data_type)

        if not os.path.exists(final_path):
            os.makedirs(final_path)
        # else:
        #     if len(os.listdir(final_path)) == 

        manifest = []
        for i in tqdm(range(len(img_dict))):
            annIds = self.coco.getAnnIds(imgIds=img_dict[i]["id"], catIds=catIds,
                                    iscrowd=False)

            anns = self.coco.loadAnns(annIds)

            threshold = False
            bboxes = []
            scale_x = size[0] / img_dict[i]["width"]
            scale_y = size[1] / img_dict[i]["height"]
            
            for ann in anns:
                # print(ann['area'])
                if ann['area'] >= min_area:
                    threshold = True
                    x, y, w, h = ann['bbox']
                    bboxes.append([int(x * scale_x), int(y * scale_y), int(w*scale_x), int(h*scale_y)])


            if threshold:
                file_name = img_dict[i]["file_name"]
                img = cv2.imread(os.path.join(coco_path, file_name))
                img = cv2.resize(img, size, cv2.INTER_AREA)
                cv2.imwrite(os.path.join(final_path, file_name), img)

                manifest.append({'category': category, 'file_name': os.path.join(final_path, file_name), 'bboxes': bboxes, 'scale':(scale_x, scale_y)})
        # print("non_one_anns: ", num_anns)
        if manifest_path is None:
            manifest_path = './manifests'
            os.makedirs(manifest_path)
        elif not os.path.exists(manifest_path):
            os.makedirs(manifest_path)
            
        with open(f'{manifest_path}/{self.data_type}_manifest_{category}.pkl', 'wb') as f:
            pickle.dump(manifest, f)
    