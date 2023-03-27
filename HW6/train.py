import torch
import torch.nn as nn
import torch.optim as optim
import csv
from tqdm import tqdm


def train(train_loader, ynet, epochs=2, net_name="ynet"):
    device = "cuda" if torch.cuda.is_available() == True else "cpu"
    print(device)
    criterion1 = nn.BCELoss()
    criterion2 = nn.MSELoss()
    criterion3 = nn.CrossEntropyLoss()
    sigmoid = nn.Sigmoid()
    softmax = nn.Softmax(dim=1)

    optimizer = optim.Adam(ynet.parameters(), lr=1e-4)  ## (6)

    logger = open("./solutions/test.csv", "a", newline="")
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

            present_obj = torch.nonzero(true_yolo_aug[:, :, :, 0])

            pred_regression_box = out[
                present_obj[:, 0], present_obj[:, 1], present_obj[:, 2], 1:5
            ]
            true_regression_box = true_yolo_aug[
                present_obj[:, 0], present_obj[:, 1], present_obj[:, 2], 1:5
            ]

            pred_labels = out[
                present_obj[:, 0], present_obj[:, 1], present_obj[:, 2], 5:-1
            ]
            true_labels = true_yolo_aug[
                present_obj[:, 0], present_obj[:, 1], present_obj[:, 2], 5:-1
            ]

            loss_BCE = criterion1(sigmoid(out[:, :, :, 0]), true_yolo_aug[:, :, :, 0])

            loss_MSE = criterion2(pred_regression_box, true_regression_box)

            loss_CE = criterion3(pred_labels, torch.argmax(true_labels, dim=1))

            total_loss = loss_BCE + loss_MSE + loss_CE
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

            if i % 100 == 0:
                logger = open("./solutions/test.csv", "a", newline="")
                data = [
                    epoch + 1,
                    i + 1,
                    loss_BCE.item(),
                    loss_MSE.item(),
                    loss_CE.item(),
                    running_loss / 100,
                ]
                with logger:
                    write = csv.writer(logger)
                    write.writerow(data)

                if running_loss < loss_flag:
                    loss_flag = running_loss

                    torch.save(ynet.state_dict(), "./solutions/" + net_name + ".pt")
                running_loss = 0.0