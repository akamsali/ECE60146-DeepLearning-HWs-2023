from DLStudio import DLStudio

import torch
import torch.nn as nn
from torch import optim

import csv


class Train:
    def __init__(
        self,
        dataroot,
        saved_path="./solutions",
        momentum=0.9,
        learning_rate=1e-5,
        epochs=1,
        batch_size=1,
        path_to_saved_embeddings = "/Users/akshita/Documents/Acads/data/word2vec"
    ) -> None:
        self.dls = DLStudio(
            dataroot=dataroot,
            path_saved_model=saved_path,
            momentum=momentum,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            classes=("negative", "positive"),
            use_gpu=True if torch.cuda.is_available() else False,
        )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() 
                                   else "cpu")
        self.path_to_saved_embeddings = path_to_saved_embeddings

        

    def get_dataloaders(self, data_size):
        text_cl = DLStudio.TextClassificationWithEmbeddings( dl_studio = self.dls )
        dataset_archive_train = f"sentiment_dataset_train_{data_size}.tar.gz"
        dataset_archive_test = f"sentiment_dataset_test_{data_size}.tar.gz"
        dataserver_train = DLStudio.TextClassificationWithEmbeddings.SentimentAnalysisDataset(
                                        train_or_test = 'train',
                                        dl_studio = self.dls,
                                        dataset_file = dataset_archive_train,
                                        path_to_saved_embeddings = self.path_to_saved_embeddings,
                        )
        dataserver_test = DLStudio.TextClassificationWithEmbeddings.SentimentAnalysisDataset(
                                        train_or_test = 'test',
                                        dl_studio = self.dls,
                                        dataset_file = dataset_archive_test,
                                        path_to_saved_embeddings = self.path_to_saved_embeddings,
                        )
        text_cl.dataserver_train = dataserver_train
        text_cl.dataserver_test = dataserver_test

        text_cl.load_SentimentAnalysisDataset(dataserver_train, dataserver_test)
        
        return text_cl

    def train(self, net, name, data_size=400, lr=1e-5, momentum=0.9, epochs=1):
        text_cl = self.get_dataloaders(data_size)
        print("Done loading data")
        net = net.to(self.device)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(net.parameters(), lr = lr, betas=(momentum, 0.999))
        training_loss_tally = list()
        softmax = nn.Softmax(dim=0)
        loss_flag = 1e32
        print("Starting training")
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(text_cl.train_dataloader):
                # get a sample from the train loader
                # print(da)
                review_tensor = data['review']
                sentiment = data['sentiment']

                review_tensor = review_tensor.to(self.device)
                sentiment = sentiment.to(self.device)

                optimizer.zero_grad()
                output = net(review_tensor)
                output = softmax(output)
                loss = criterion(output, torch.argmax(sentiment))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                
                if i % 10 == 9:
                    row = [epoch, i, running_loss / 10]
                    with open(f'./solutions/{name}_training_log.csv', 'a') as csvFile:
                        writer = csv.writer(csvFile)
                        writer.writerow(row)
                    if running_loss < loss_flag:
                        loss_flag = running_loss
                        torch.save(net.state_dict(), f'./solutions/{name}_best_model.pt')
                        
                    running_loss = 0.0

           
