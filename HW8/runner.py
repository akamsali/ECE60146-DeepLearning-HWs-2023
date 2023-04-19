from train import Train
from model import RNN_custom, UNI_GRU, BI_GRU

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
# mac
# dataroot = "/Users/akshita/Documents/Acads/data/sentiment_analysis/"
# path_to_saved_embeddings = "/Users/akshita/Documents/Acads/data/word2vec"

#vm
dataroot = "/mnt/cloudNAS4/akshita/Documents/datasets/sentiment_analysis/"
path_to_saved_embeddings = "/mnt/cloudNAS4/akshita/Documents/datasets/word2vec"

train = Train(dataroot, path_to_saved_embeddings=path_to_saved_embeddings)
hidden_size = 512
net = RNN_custom(300, hidden_size, 2, True, 2)
lr = 1e-5
momentum = 0.9
name = f"RNN_custom_{hidden_size}_{str(lr).replace('.', '_')}_{str(momentum).replace('.', '_')}"
train.train_mygru(net, name, data_size=400, lr=lr, momentum=momentum, epochs=5, batch_size=1)

train_loader = train.get_dataloaders(400).train_loader
test_loader = train.get_dataloaders(400).test_loader

net_uni = UNI_GRU(300, 512, 2, 1)
net_bi = BI_GRU(300, 512, 2, 1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def train(net, dataloader, name='uni', epochs=1):
    net = net.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-5, betas=(0.9, 0.999))
    print("training")
    running_loss = 0.0
    for epoch in range(epochs):
        for i, data in tqdm(enumerate(dataloader)):
            review_tensor = data['review']
            sentiment = data['sentiment']
            review_tensor = review_tensor.to(device)
            sentiment = sentiment.to(device)

            optimizer.zero_grad()
            output, _ = net(review_tensor)
            loss = criterion(output, torch.argmax(sentiment, dim=1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                row = [epoch, i, running_loss / 100]
                with open(f'./solutions/{name}_training_log.csv', 'a') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(row)
                    
                running_loss = 0.0
    torch.save(net.state_dict(), f'./solutions/{name}_model.pt')

train(net_uni, train_loader, name='uni', epochs=5)
train(net_bi, train_loader, name='bi', epochs=5)
