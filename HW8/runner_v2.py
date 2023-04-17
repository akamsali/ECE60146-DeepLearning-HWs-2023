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

class MyGRU(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 bias=True):
        """
        Args:
            input_size: size of input vectors
            hidden_size: size of hidden state vectors
            bias: whether to use bias parameters or not
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

        self.reset_parameters()


    def reset_parameters(self):
        # Initialize all weights uniformly in the range [-1/sqrt(n), 1/sqrt(n)]
        # n = hidden_size
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx=None):
        """
        Args:
            input: of shape (batch_size, input_size)
            hx: of shape (batch_size, hidden_size)

        Returns:    
            hy: of shape (batch_size, hidden_size)
        """
        
        if hx is None:
            hx = Variable(input.new_zeros(input.size(0), self.hidden_size))

        # Compute x_t and h_t
        x_t = self.x2h(input)
        h_t = self.h2h(hx)

        # we split the output of the linear layers into 3 parts
        # each of size hidden_size
        x_reset, x_upd, x_new = x_t.chunk(3, 1)
        h_reset, h_upd, h_new = h_t.chunk(3, 1)

        # compute the reset, update and new gates
        reset_gate = torch.sigmoid(x_reset + h_reset)
        update_gate = torch.sigmoid(x_upd + h_upd)
        new_gate = torch.tanh(x_new + (reset_gate * h_new))

        hy = update_gate * hx + (1 - update_gate) * new_gate

        return hy
    
class RNN_custom(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size,
                 num_layers, 
                 bias,
                output_size
                ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.output_size = output_size

        # list of GRU cells
        self.rnn_list = nn.ModuleList()
        self.rnn_list.append(MyGRU(input_size, 
                                     hidden_size, 
                                     bias))
        for i in range(num_layers-1):
            self.rnn_list.append(MyGRU(hidden_size, 
                                         hidden_size, 
                                         bias))
        # feedforward layer
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input, hx=None):
        """
        Args:
            input: of shape (batch_size, seq_len, input_size)
            hx: of shape (batch_size, hidden_size)

        Returns:
            output: of shape (batch_size, output_size)
        """
        batch_size, seq_len, _ = input.shape

        if hx is None:
            h0 = Variable(torch.zeros(self.num_layers,
                                      batch_size,
                                      self.hidden_size))
        else:
            h0 = hx
        
        # list of hidden states
        h_list = []
        outs = []
        for i in range(self.num_layers):
            h_list.append(h0[i, :, :])
        
        # for each time step
        for t in range(seq_len):
            for i in range(self.num_layers):
                if i == 0:
                    h_l = self.rnn_list[i](input[:, t, :], h_list[i])
                else:
                    h_l = self.rnn_list[i](h_list[i-1], h_list[i])
                h_list[i] = h_l
            outs.append(h_l)
        # feedforward layer
        output = self.linear(outs[-1].squeeze(0))
        return output
    
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
    training_loss_tally = list()
    softmax = nn.Softmax(dim=0)
    loss_flag = 1e32
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
            
            if i % 10 == 9:
                row = [epoch, i, running_loss / 10]
                with open(f'./solutions/{name}_training_log.csv', 'a') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(row)
                if running_loss < loss_flag:
                    loss_flag = running_loss
                    torch.save(net.state_dict(), f'./solutions/{name}_best_model.pt')
                    
                running_loss = 0.0
    # return training_loss_tally

net = RNN_custom(300, 512, 2, True, 2)
train_with_my_gru(net, 'my_gru', device)