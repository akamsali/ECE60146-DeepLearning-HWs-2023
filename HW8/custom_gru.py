from DLStudio import DLStudio

import torch
from torch import optim
from torch import nn as nn 
from torch.autograd import Variable

import numpy as np
import csv

dataroot = "/mnt/cloudNAS4/akshita/Documents/datasets/sentiment_analysis/"
path_to_saved_embeddings = "/mnt/cloudNAS4/akshita/Documents/datasets/word2vec"
dataset_archive_train = "sentiment_dataset_train_400.tar.gz"
dataset_archive_test = "sentiment_dataset_test_400.tar.gz"

#taken from Avi's code
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


def train_with_my_gru(net, name, device):
    net = net.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters(), lr = dls.learning_rate, betas=(dls.momentum, 0.999))
    softmax = nn.Softmax(dim=0)
    for epoch in range(dls.epochs):
        running_loss = 0.0
        for i, data in enumerate(text_cl.train_dataloader):
            # get a sample from the train loader
            review_tensor = data['review']
            sentiment = data['sentiment']

            review_tensor = review_tensor.to(device)
            sentiment = sentiment.to(device)

            optimizer.zero_grad()
            output = net(review_tensor)
            output = softmax(output)
            loss = criterion(output, torch.argmax(sentiment))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if i % 100 == 99:
                row = [epoch, i, running_loss / 100]
                with open(f'./solutions/{name}_training_log.csv', 'a') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(row)
                    
                running_loss = 0.0
    
    torch.save(net.state_dict(), f'./solutions/{name}_best_model.pt')
    # return training_loss_tally

net = RNN_custom(300, 512, 1, True, 2)
train_with_my_gru(net, 'my_gru', device)