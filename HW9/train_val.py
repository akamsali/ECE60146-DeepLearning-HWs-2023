import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import numpy as np

# desktop



def train(net, train_loader, epochs=1, name="test"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net = net.to(device)
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            # print(outputs.shape, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                row_info = [epoch + 1, i + 1, running_loss / 100]
                with open(name + "_train.csv", "a") as f:
                    writer = csv.writer(f)
                    writer.writerow(row_info)

        torch.save(net.state_dict(), name + "_epoch_" + str(epoch + 1) + ".pt")
        print("Finished Training Epoch {}".format(epoch + 1))


def test(net, val_loader, name="test"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net = net.to(device)
    net.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            # labels = labels.to(device)
            outputs = net(images)
            outputs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            predictions.append(predicted.detach().cpu().numpy())
            targets.append(labels.numpy())

    # return predictions, targets
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    # print(predictions.shape, targets.shape)

    acc = accuracy_score(targets, predictions)

    with open(name + f"_val_{acc}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["predictions", "targets"])
        for i in range(len(predictions)):
            writer.writerow([predictions[i], targets[i]])

    cm = confusion_matrix(targets, predictions)
    # print(cm)
    return cm, acc