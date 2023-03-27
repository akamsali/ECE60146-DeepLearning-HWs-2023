from coco_custom import COCO_loader
from dataset import MyDataset
from model import NetForYolo
from train import train
from val import validation

from torch.utils.data import DataLoader
import torch
import cv2
import pickle
import numpy as np

categories = ["bus", "cat", "pizza"]

# create custom dataset
train_cl = COCO_loader()
val_cl = COCO_loader(dataType='val')

for category in categories:
    train_cl.save_images_to_folder(category, min_area=4096)
    val_cl.save_images_to_folder(category, min_area=4096)

# plot images
num_images = 3
for data_split in ["train", "val"]:
    all_images = []
    for category in categories:

        split_path = f"./manifests/{data_split}_manifest_{category}.pkl"
        with open(split_path, "rb") as handle:
            data = pickle.load(handle)
        
        cat_count = 0
        cat_images = []
        for d in data:
            if len(d["bboxes"]) > 2 and cat_count < 3:
                img = cv2.imread(d["file_name"])
                for bbox in d['bboxes']:
                    x, y, w, h = bbox
                    img = cv2.rectangle(
                        img, (x, y), (x + w, y + h), color=(36, 255, 0), thickness=2
                    )
                    img = cv2.putText(
                        img,
                        org=(x, y - 10),
                        text=d["category"],
                        color=(255, 0, 0),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.8,
                        thickness=2,
                    )

                cat_count += 1
            
                cat_images.append(img)
        
        all_images.append(np.concatenate(cat_images, axis=1))

    all_images = np.concatenate(all_images, axis=0)

    cv2.imwrite(f"./solutions/{data_split}_sampled_images.png", all_images)


# instantiate model
ynet = NetForYolo(depth=2)
batch_size = 8 # set batch size
# train_data = MyDataset(
#     categories=categories, split="train", manifest_path="./manifests", mac=True
# )
# create train dataloader
train_data = MyDataset(
    categories=categories, split="train", manifest_path="./manifests"
)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# train 
train(train_loader, ynet, epochs=10, net_name='final_train')

# load weights from checkpoint
y_params = torch.load('/Users/akshita/Documents/Acads/models/final_train.pt', map_location=torch.device('cpu'))
ynet.load_state_dict(y_params)

# create validation dataloader
val_data = MyDataset(
    categories=categories, split="val", manifest_path="./manifests", mac=True
)
val_data = MyDataset(categories=categories, split="val", manifest_path="./manifests")
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
validation(val_loader, ynet)

from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def validate_and_conf_matrix(t, p, categories, name="Net") -> None:
    cm = confusion_matrix(t, p)
    plt.figure()
    sns.heatmap(cm, annot=cm, xticklabels=categories, yticklabels=categories, fmt="g")
    plt.title(f"Confusion matrix for {name}, accuracy={accuracy_score(t, p)}")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.savefig(f"./solutions/cm_{name}.png")

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

t, p = [], []
iou_tot = {}
bbx_tot = {}
labels = {}
with open("./pkl_files/predicted_bboxes.pkl", 'rb') as handle:
    predicted_bboxes_np = pickle.load(handle)

with open("./pkl_files/total_true_bboxes.pkl", 'rb') as handle:
    total_true_bboxes = pickle.load(handle)
total_true_bboxes_np = []
for i in range(len(total_true_bboxes)):
    total_true_bboxes_np.append(np.array(total_true_bboxes[i]))

with open("./pkl_files/total_val_classes.pkl", 'rb') as handle:
    total_val_classes = pickle.load(handle)

with open("./pkl_files/true_val_classes.pkl", 'rb') as handle:
    true_val_classes = pickle.load(handle)

# print(len(predicted_bboxes_np))
for i in range(len(predicted_bboxes_np)):
    iou = []
    bbx = []
    for j in range(total_true_bboxes_np[i].shape[0]):
        # print(predicted_bboxes_np[i][:, j])
        t_x, t_y, t_w, t_h = total_true_bboxes[i][j]
        x, y, w, h = predicted_bboxes_np[i][:, j]
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        if ((w < 0 or h < 0) or (x+w > 256 or y+h > 256)
            or ((w < 64) and (h < 64))
            ):
            continue

        iou.append(bb_intersection_over_union([x, y, x+w, y+h], 
                                                 [t_x, t_y, t_x+t_w, t_y+t_h]))
        bbx.append(([x, y, x+w, y+h], [t_x, t_y, t_x+t_w, t_y+t_h]))
        # print(total_val_classes[i][j], true_val_classes[i])
        t.append(total_val_classes[i][j])
        p.append(true_val_classes[i])
    if len(iou):
        iou_tot[i] = iou
        bbx_tot[i] = bbx
        labels[i] = true_val_classes[i]
    
validate_and_conf_matrix(t, p, categories, name="yolo_net")



flag = 0
for i in list(iou_tot.keys()):
    # cnt = 0
    img = cv2.imread(val_data.clip_file_names(val_data.file_list[i]))
    for j in range(len(iou_tot[i])):
        # various conditions can be used to filter out the bounding boxes
        # if iou_tot[i][j] == 0 : # > 0.5:  
        #     cnt += 1
        # print(i, cnt)
            # print("True")
            # print(i, j)
            img = cv2.rectangle(img, (bbx_tot[i][j][0][0], bbx_tot[i][j][0][1]),
                                (bbx_tot[i][j][0][2], bbx_tot[i][j][0][3]),
                                color=(0, 0, 255), thickness=2)
            
            img = cv2.rectangle(img, (bbx_tot[i][j][1][0], bbx_tot[i][j][1][1]),
                                (bbx_tot[i][j][1][2], bbx_tot[i][j][1][3]),
                                color=(0, 255, 0), thickness=2)
            
            img = cv2.putText(img, f"{categories[labels[i]]}", (bbx_tot[i][j][0][0], bbx_tot[i][j][0][1]-10),
                        color=(255, 0, 0), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)
    # this condition can be used to save the images with more than one bounding box
    # if cnt > 1: cv2.imwrite(f"./{i}_true.png", img)
    
    cv2.imwrite(f"./{i}_true.png", img)