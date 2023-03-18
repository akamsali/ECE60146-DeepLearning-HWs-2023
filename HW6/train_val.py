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


def validation(val_loader, ynet, yolo_interval=32):
    device = "cuda" if torch.cuda.is_available() == True else "cpu"
    print(device)
    ynet = ynet.to(device)
    ynet.eval()
    predicted_bboxes = []
    total_true_bboxes = []
    total_val_classes = []
    true_val_classes = []
    with torch.no_grad():
        for data in val_loader:
            inp, t_l, true_bboxes = data
            inp = inp.to(device)

            out = ynet(inp)
            out = out.view(-1, 64, 5, 9)

            # get cells with top 5 highest values in the first element of the predicted yolo_vectors
            # to achieve this, first we get the max value of anchor boxes for each cell
            # then we sort the values in descending order and get the top 5 cells
            pred_vals = out[:, :, :, 0]
            pred_vals, _ = torch.max(pred_vals, dim=-1)
            sorted_cells = torch.argsort(pred_vals, descending=True, dim=-1)

            top5preds = torch.zeros((out.shape[0], 64, 9))

            for i in range(out.shape[0]):
                each_batch = out[i, sorted_cells[i]]
                for j in range(each_batch.shape[0]):
                    temp = each_batch[j, :, 0]
                    args = torch.argmax(temp, dim=-1)
                    top5preds[i, j] = each_batch[j, args]

            pred_classes = top5preds[:, :, 5:-1]
            pred_classes = nn.Softmax(dim=1)(pred_classes)
            pred_classes = torch.argmax(pred_classes, dim=-1)

            pred_regression_vec = top5preds[:, :, 1:5]
            del_x, del_y = pred_regression_vec[:, :, 0], pred_regression_vec[:, :, 1]
            h, w = pred_regression_vec[:, :, 2], pred_regression_vec[:, :, 3]

            h *= yolo_interval
            w *= yolo_interval
            cell_row_index = torch.div(sorted_cells, 8, rounding_mode="floor")
            cell_col_index = sorted_cells % 8

            yolo_offset = torch.ones_like(cell_row_index) * (yolo_interval / 2)
            bb_center_x = (
                cell_col_index * yolo_interval + yolo_offset + del_x * yolo_interval
            )
            bb_center_y = (
                cell_row_index * yolo_interval + yolo_offset + del_y * yolo_interval
            )

            bb_top_left_x = bb_center_x - torch.div(w, 2, rounding_mode="floor")
            bb_top_left_y = bb_center_y - torch.div(h, 2, rounding_mode="floor")

            valid_preds = []
            val_classes = []
            for i in range(bb_top_left_x.shape[0]):
                for j in range(bb_top_left_x.shape[1]):
                    if bb_top_left_x[i, j] < 0:
                        bb_top_left_x[i, j] = 0
                    if bb_top_left_y[i, j] < 0:
                        bb_top_left_y[i, j] = 0
                    if (
                        (h[i, j] > 256).any()
                        or (w[i, j] > 256).any()
                        or (w < 64).any()
                        or (h < 64).any()
                    ):
                        continue
                    valid_preds.append(
                        torch.tensor(
                            [bb_top_left_x[i, j], bb_top_left_y[i, j], w[i, j], h[i, j]]
                        )
                    )
                    val_classes.append(pred_classes[i, j])

            if len(valid_preds):
                valid_preds = torch.stack(valid_preds, dim=-1)
                predicted_bboxes.append(valid_preds)
                total_true_bboxes.append(true_bboxes)
                total_val_classes.append(val_classes)
                true_val_classes.append(t_l)

        if predicted_bboxes:
            predicted_bboxes = torch.cat(predicted_bboxes, dim=-1)

        return predicted_bboxes, total_true_bboxes, total_val_classes, true_val_classes
