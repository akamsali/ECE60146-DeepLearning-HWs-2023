from DLStudio import DLStudio

import torch

import torch.nn as nn
from torch import optim
import tqdm
import csv

#taken from Avi's code


dataroot = "/mnt/cloudNAS4/akshita/Documents/datasets/sentiment_analysis/"
path_to_saved_embeddings = "/mnt/cloudNAS4/akshita/Documents/datasets/word2vec"
dataset_archive_train = "sentiment_dataset_train_400.tar.gz"
dataset_archive_test = "sentiment_dataset_test_400.tar.gz"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dls = DLStudio(
                  dataroot = dataroot,
                  path_saved_model = "./saved_model",
                  momentum = 0.9,
                  learning_rate =  1e-5,
                  epochs = 1,
                  batch_size = 1,
                  classes = ('negative','positive'),
                  use_gpu = True if torch.cuda.is_available() else False,
              )

text_cl = DLStudio.TextClassificationWithEmbeddings( dl_studio = dls )

dataserver_train = DLStudio.TextClassificationWithEmbeddings.SentimentAnalysisDataset(
                                 train_or_test = 'train',
                                 dl_studio = dls,
                                 dataset_file = dataset_archive_train,
                                 path_to_saved_embeddings = path_to_saved_embeddings,
                   )
dataserver_test = DLStudio.TextClassificationWithEmbeddings.SentimentAnalysisDataset(
                                 train_or_test = 'test',
                                 dl_studio = dls,
                                 dataset_file = dataset_archive_test,
                                 path_to_saved_embeddings = path_to_saved_embeddings,
                  )
text_cl.dataserver_train = dataserver_train
text_cl.dataserver_test = dataserver_test

text_cl.load_SentimentAnalysisDataset(dataserver_train, dataserver_test)

class UNI_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1) -> None:
        super(UNI_GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax()
    
    def forward(self, x):
        h = torch.zeros(self.num_layers, x.size(1), self.hidden_size).requires_grad_().cuda()
        # Forward propagation by passing in the input and hidden state into the model
        out, h = self.gru(x, h.detach())
        # print(out.shape)
        out = self.fc(self.relu(out[:, -1]))
        out = self.logsoftmax(out)
        return out, h
    

class BI_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1) -> None:
        super().__init__()
        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, output_size)
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        h = torch.zeros(2*self.num_layers, x.size(1), self.hidden_size).requires_grad_().cuda()
        # print(h.shape)
        out, h = self.gru(x,h)
        print(out.shape, out[:, -1].shape)
        out = self.fc(self.relu(out[:, -1]))
        out = self.logsoftmax(out)
        return out, h
    
net_uni = UNI_GRU(300, 512, 2, 1)
net_bi = BI_GRU(300, 512, 2, 1)

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

train(net_uni, text_cl.train_dataloader, name='uni', epochs=1)
train(net_bi, text_cl.train_dataloader, name='bi', epochs=1)
