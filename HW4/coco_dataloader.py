from pycocotools.coco import COCO
import skimage.io as io
from skimage.transform import resize

import os


class COCO_loader:
    def __init__(self, dataDir='/mnt/cloudNAS4/jo/coco', 
                    dataType='train2014', 
                    ) -> None:

        self.data_dir = dataDir
        self.data_type = dataType
    
        self.annFile='{}/annotations/instances_{}.json'.format(self.data_dir,self.data_type)
        # initialize COCO api for instance annotations
        self.coco=COCO(self.annFile)
        self.cats = self.coco.loadCats(self.coco.getCatIds())

    def display_cartegories(self):
        nms=[cat['name'] for cat in self.cats]
        print('COCO categories: \n{}\n'.format(' '.join(nms)))

    # display COCO categories and supercategories
    def display_supercats(self):
        nms = set([cat['supercategory'] for cat in self.cats])
        print('COCO supercategories: \n{}'.format(' '.join(nms)))

    def get_img_list(self, category, train_size, val_size):
        catIds = self.coco.getCatIds(catNms=category)
        imgIds = self.coco.getImgIds(catIds=catIds )
        img_dict = self.coco.loadImgs(ids=imgIds)
        train = [img_dict[i]['file_name'] for i in range(train_size)]
        val = [img_dict[i]['file_name'] for i in range(train_size, train_size+val_size)]

        return train, val
    
    def save_images_to_folder(self, category, train_size=5, val_size=5,
                              data_path='/mnt/cloudNAS4/akshita/Documents/datasets/coco_custom'):

        path = os.path.join(data_path, category)

        if os.path.exists(os.path.join(path, 'train')) \
            and os.path.exists(os.path.join(path, 'val')) \
            and len(os.listdir(os.path.join(path, 'train'))) == train_size \
            and len(os.listdir(os.path.join(path, 'val'))) == val_size:

            print(f"{category} already present")

        else:
            if not os.path.exists(os.path.join(path, 'train')):
                os.makedirs(os.path.join(path, 'train'))
            if not os.path.exists(os.path.join(path, 'val')):
                os.makedirs(os.path.join(path, 'val'))

            train_list, val_list = self.get_img_list(category=category, 
                                                     train_size=train_size, 
                                                     val_size=val_size)
            
            
            coco_path = os.path.join(self.data_dir, 'images')
            train_path = os.path.join(path, 'train')
            val_path = os.path.join(path, 'val')
            
            for tf in train_list:
                img = io.imread(os.path.join(coco_path, tf))
                img = resize(img, (64,64))
                io.imsave(os.path.join(train_path, tf), img)
            for vf in val_list:
                img = io.imread(os.path.join(coco_path, vf))
                img = resize(img, (64,64))
                io.imsave(os.path.join(val_path, vf), img) 