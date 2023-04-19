import numpy as np

import torch
import torch.nn as nn


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
        # Repeat the same linear layer 3 times for the reset, update and new gates
        # The input to the linear layer is the input vector x_t
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        # The input to the linear layer is the hidden state vector h_t
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
            hx = torch.zeros(input.size(0), self.hidden_size)

        # Compute x_t and h_t
        x_t = self.x2h(input)
        h_t = self.h2h(hx)

        # we split the output of the linear layers into 3 parts
        # each of size hidden_size, each split is for a gate (reset, update, new)
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
            if torch.cuda.is_available():
                h0 = torch.zeros(self.num_layers,
                                          batch_size,
                                          self.hidden_size).cuda().requires_grad_()
            else:

                h0 = torch.zeros(self.num_layers,
                                          batch_size,
                                          self.hidden_size).requires_grad_()
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
        h = torch.zeros(self.num_layers, x.size(1), self.hidden_size).cuda().requires_grad_()
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
        h = torch.zeros(2*self.num_layers, x.size(1), self.hidden_size).cuda().requires_grad_()
        # print(h.shape)
        out, h = self.gru(x,h)
        print(out.shape, out[:, -1].shape)
        out = self.fc(self.relu(out[:, -1]))
        out = self.logsoftmax(out)
        return out, h