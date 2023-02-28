from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F


from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
import csv

from typing import Union


def train(
    net, train_dataloader, epochs=10, net_name="net_name"
) -> None:
    device = "cuda" if torch.cuda.is_available() == True else "cpu"
    net.train()
    net = net.to(device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.99))

    # tot_loss = []
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        loss_flag = 1e32
        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                # opening the csv file in 'w+' mode
                # print("[epoch]: %d, batch: %5d] loss: %.3f" %(epoch+1, i+1, running_loss / 100))
                data = [epoch + 1, i + 1, running_loss / 100]
                file = open("results/" + net_name + ".csv", "a", newline="")
                # writing the data into the file
                with file:
                    write = csv.writer(file)
                    write.writerow(data)

                # tot_loss.append(running_loss/100)
                if running_loss < loss_flag:
                    loss_flag = running_loss
                    torch.save(net.state_dict(), "results/" + net_name + ".pt")
                running_loss = 0.0


def validation(net, val_dataset) -> Union[list, list]:
    true_labels = []
    pred_labels = []
    net.eval()
    for i in range(len(val_dataset)):
        img, label = val_dataset.__getitem__(i)
        # print(img.shape)
        true_labels.append(label)
        output = net(img.unsqueeze(0))
        output = F.softmax(output, dim=1)
        output = torch.argmax(output)
        pred_labels.append(output.item())
        # break
    return true_labels, pred_labels


def validate_and_conf_matrix(net, val_dataset, categories, name="Net") -> None:
    t, p = validation(net, val_dataset=val_dataset)
    cm = confusion_matrix(t, p)
    plt.figure()
    sns.heatmap(cm, annot=cm, xticklabels=categories, yticklabels=categories, fmt="g")
    plt.title(f"Confusion matrix for {name}, accuracy={accuracy_score(t, p)}")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.savefig(f"results/cm_{name}.png")
