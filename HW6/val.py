from dataset import MyDataset
# from torch.utils.data import DataLoader
from model import NetForYolo
import torch 
import torch.nn as nn
from tqdm import tqdm


categories = ["bus", "cat", "pizza"]
saved_model = '/Users/akshita/Documents/Acads/models/final_train.pt'
# saved_model = './solutions/final_train.pt'

ynet = NetForYolo(depth=2)
y_params = torch.load(saved_model, map_location=torch.device('cpu'))
ynet.load_state_dict(y_params)


# batch_size = 4
# train_data = MyDataset(categories=categories, split='train', manifest_path='./manifests', mac=True)
val_data = MyDataset(categories=categories, split='val', manifest_path='./manifests', mac=True)
# val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

# next(iter(val_loader))

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
        # for data in val_loader:
        for d in tqdm(range(len(val_loader))):
        # for d in tqdm(range(10)):
            inp, t_l, true_bboxes = val_data.__getitem__(d)
            # data
        #     inp, t_l, true_bboxes = data
            inp = inp.to(device)

            out = ynet(inp.unsqueeze(0))
            out = out.view(-1, 64, 5, 9)

        #     # get cells sorted with the highest values in the first element of the predicted yolo_vectors
        #     # to achieve this, first we get the max value of anchor boxes for each cell
        #     # then we sort the values in descending order
            pred_vals = out[:, :, :, 0]
            pred_vals, _ = torch.max(pred_vals, dim=-1)
            sorted_cells = torch.argsort(pred_vals, descending=True, dim=-1)

            # toppred_iter = torch.zeros((out.shape[0], 64, 9))

            # for i in range(out.shape[0]):
            #     each_batch = out[i, sorted_cells[i]]
            #     for j in range(each_batch.shape[0]):
            #         temp = each_batch[j, :, 0]
            #         args = torch.argmax(temp, dim=-1)
            #         toppred_iter[i, j] = each_batch[j, args]

            toppred = out[sorted_cells[:, 0], sorted_cells[:, 1], sorted_cells[:2]]
            # print(toppred, toppred_iter)
            pred_classes = toppred[:, :, 5:-1]
            pred_classes = nn.Softmax(dim=1)(pred_classes)
            pred_classes = torch.argmax(pred_classes, dim=-1)

            pred_regression_vec = toppred[:, :, 1:5]
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
                        # or ((w < 64).any() and (h < 64).any())
                    ):
                        # print(d)
                        continue
                    valid_preds.append(
                        torch.tensor(
                            [bb_top_left_x[i, j], bb_top_left_y[i, j], w[i, j], h[i, j]]
                        )
                    )
                    val_classes.append(pred_classes[i, j].item())

            if len(valid_preds):
                valid_preds = torch.stack(valid_preds, dim=-1)
                predicted_bboxes.append(valid_preds)
                # print(true_bboxes)
                total_true_bboxes.append(true_bboxes)
                total_val_classes.append(val_classes)
                # print(val_classes)
                true_val_classes.append(t_l)


    return predicted_bboxes, total_true_bboxes, total_val_classes, true_val_classes
    
predicted_bboxes, total_true_bboxes, total_val_classes, true_val_classes = validation(val_data, ynet)
# predicted_bboxes.shape

import pickle

predicted_bboxes_np = []
for p in predicted_bboxes:
    predicted_bboxes_np.append(p.numpy())


with open('predicted_bboxes.pkl', 'wb') as f:
    pickle.dump(predicted_bboxes_np, f)
with open('total_true_bboxes.pkl', 'wb') as f:
    pickle.dump(total_true_bboxes, f)
with open('total_val_classes.pkl', 'wb') as f:
    pickle.dump(total_val_classes, f)
with open('true_val_classes.pkl', 'wb') as f:
    pickle.dump(true_val_classes, f)