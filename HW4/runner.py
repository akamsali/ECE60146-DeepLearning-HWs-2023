from coco_dataloader import COCO_loader
from tqdm import tqdm


categories = ['airplane', 'bus', 'cat', 'dog', 'pizza']
data_path = '/mnt/cloudNAS4/akshita/Documents/datasets/coco_custom'
train_size, val_size = 1500, 500
cl = COCO_loader()

for category in tqdm(categories):
    cl.save_images_to_folder(category=category, train_size=train_size, val_size=val_size)