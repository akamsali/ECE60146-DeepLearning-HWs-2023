from dataset import MyDataset

from model import NetForYolo

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import csv 
from tqdm import tqdm

categories = ["bus", "cat", "pizza"]

batch_size = 4
train_data = MyDataset(categories=categories, split='train', manifest_path='./manifests')

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

ynet = NetForYolo(depth=2)


device = "cuda" if torch.cuda.is_available() == True else "cpu"
print(device)
criterion1 = nn.BCELoss()                    # For the first element of the 8 element yolo vector              ## (3)
criterion2 = nn.MSELoss()                    # For the regression elements (indexed 2,3,4,5) of yolo vector   ## (4)
criterion3 = nn.CrossEntropyLoss()           # For the last three elements of the 8 element yolo vector        ## (5)
sigmoid = nn.Sigmoid()

# print("\n\nLearning Rate: ", self.rpg.learning_rate)
optimizer = optim.Adam(ynet.parameters(), lr=1e-4)                 ## (6)
epochs = 2
# total_loss = []
logger = open('./solutions/test.csv', 'a', newline="")
loss_flag = 1e32
ynet = ynet.to(device)
for epoch in range(epochs):
    running_loss = 0
    for i, d in tqdm(enumerate(train_loader)):
        inp, t_l, true_yolo_aug = d
        optimizer.zero_grad()
        inp = inp.to(device)
        true_yolo_aug = true_yolo_aug.to(device)

        out = ynet(inp)
        out = out.view(-1, 64, 5, 9)
        # print(out.shape)
        # pred_objectness = out[:, :, :, 0]
        # regression_box = out[:, :, :, 1:5]
        # pred_labels = out[:, :, :, 5:-1]
        # print(objectness.shape)

        # print(true_objectness.shape)
        # present_obj = torch.where(true_yolo_aug[:, :, :, 0])#.unsqueeze(0)
        present_obj = torch.nonzero(true_yolo_aug[:, :, :, 0])#.unsqueeze(0)

        # print(present_obj)
        #pred_objectness = torch.zeros((len(present_obj)))
        #true_objectness = torch.ones((len(present_obj)))


        pred_regression_box = torch.zeros((len(present_obj), 4))
        true_regression_box = torch.zeros((len(present_obj), 4))

        pred_labels = torch.zeros((len(present_obj), 3))
        true_labels = torch.zeros((len(present_obj), 3))

        for j, p in enumerate(present_obj):
            # pred_objectness[i] = out[p[0], p[1], p[2], 0]
            # print(p)
            pred_regression_box[j] = out[p[0], p[1], p[2], 1:5]
            true_regression_box[j] = true_yolo_aug[p[0], p[1], p[2], 1:5]

            pred_labels[j] = out[p[0], p[1], p[2], 5:-1]
            true_labels[j] = true_yolo_aug[p[0], p[1], p[2], 5:-1]
        #print(pred_labels, true_labels, t_l)
        #break
        loss_BCE = criterion1(sigmoid(out[:, :, : , 0]), true_yolo_aug[:, :, :, 0])
        # print(loss_BCE)

        loss_MSE = criterion2(pred_regression_box, true_regression_box)
        # print(loss_MSE)

        loss_CE = criterion3(pred_labels, true_labels)
        # print(loss_CE)

        total_loss = loss_BCE + loss_MSE + loss_CE
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

        if i % 100 == 0:
 
            logger = open('./solutions/test.csv', 'a', newline="")           
            data = [epoch + 1, i + 1, loss_BCE.item(), loss_MSE.item(), loss_CE.item(), running_loss / 100]
            with logger:
                write = csv.writer(logger)
                write.writerow(data)

            if running_loss < loss_flag:
                loss_flag = running_loss

                torch.save(ynet.state_dict(), "./solutions/" + 'net_name' + ".pt")
            running_loss = 0.0
