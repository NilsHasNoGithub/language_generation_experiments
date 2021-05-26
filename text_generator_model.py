import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import sys
from sklearn.model_selection import train_test_split


class TextModel(nn.Module):

    def __init__(self, n_in, n_out, n_hidden=500):
        super().__init__()

        n1 = int(np.rint(abs(n_in - n_hidden) * (2/3) + min([n_in, n_hidden])))
        n2 = int(np.rint(abs(n_in - n_hidden) * (1/3) + min([n_in, n_hidden])))

        if n_hidden > n2:
            n2, n1 = n1, n2

        nlr = 'tanh'

        self.to_in1 = nn.LSTM(n_in, n1, batch_first=True)
        self.to_in2 = nn.LSTM(n1, n2, batch_first=True)
        self.to_in3 = nn.LSTM(n2, n_hidden, batch_first=True)

        self.ins = [self.to_in1, self.to_in2, self.to_in3]

        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_hidden)

        self.fcs = [self.fc1, self.fc2, self.fc3]

        self.to_out1 = nn.LSTM(n_hidden, n2, batch_first=True)
        self.to_out2 = nn.LSTM(n2, n1, batch_first=True)
        self.to_out3 = nn.LSTM(n1, n_out, batch_first=True)

        self.outs = [self.to_out1, self.to_out2, self.to_out3]

        self.to_out4 = nn.Linear(n_out, n_out)
        self.softmax = nn.Softmax(dim=2)

        self.relu = nn.ReLU()

        self.n_hs = len(self.ins) + len(self.outs)

        self.n_in = n_in
        self.n_out = n_out

    def forward(self, x, hss=None, return_hs=False):
        if hss is None:
            hss = [None] * self.n_hs
        i_hs = 0
        for l in self.ins:
            x, hss[i_hs] = l(x, hss[i_hs])
            i_hs += 1
        for l in self.fcs:
            x = self.relu(l(x))
        for l in self.outs:
            x, hss[i_hs] = l(x, hss[i_hs])
            i_hs += 1
        x = self.to_out4(x)
        if return_hs:
            return x, hss
        return x


# class TextModel(nn.Module):

#     def __init__(self, n_in, n_out, n_hidden=5):
#         super().__init__()
#         self.fc_in = nn.Linear(n_in, n_hidden)
#         self.rnn = nn.RNN(n_hidden, n_hidden, batch_first=True)
#         self.fc = nn.Linear(n_hidden, n_out)
#         self.relu = nn.ReLU()
#         self.softmax = nn.Softmax(dim=2)

#     def forward(self, x, hs=None):
#         return_hs = hs is not None
#         x = self.fc_in(x)
#         x = self.relu(x)
#         x, hs = self.rnn(x, hs)
#         x = self.fc(x)
#         if return_hs:
#             return x, hs
#         return x


class ConvolutionalTextModel(nn.Module):

    def __init__(self,
                 n_in, n_out,
                 n_hidden_sqrt=40,
                 n_rec_layers=1,
                 ):
        super().__init__()

        if n_hidden_sqrt % 4 != 0:
            raise Exception("n_hidden_sqrt must be divisible by 4")

        n_hidden = n_hidden_sqrt ** 2
        self.n_hidden_sqrt = n_hidden_sqrt

        # self.fc_in = nn.Linear(n_in, n_hidden)
        self.rec1 = nn.GRU(n_in, n_hidden, batch_first=True,
                           num_layers=n_rec_layers)
        self.conv1 = nn.Conv2d(1, 32, 5, stride=1, padding=2)

        conv_out_size = n_hidden_sqrt

        self.max_pool1 = nn.MaxPool2d([2, 2], stride=2, padding=0)

        conv_out_size /= 2

        self.conv2 = nn.Conv2d(32, 64, [5, 5], stride=1, padding=2)

        self.max_pool2 = nn.MaxPool2d([2, 2], stride=2, padding=0)

        conv_out_size /= 2

        self.pool_out_size = int(64 * (conv_out_size ** 2))

        n = int(abs(self.pool_out_size - n_out) /
                2 + min(self.pool_out_size, n_out))

        self.fc_out1 = nn.Linear(self.pool_out_size, n)
        self.fc_out2 = nn.Linear(n, n_out)

        self.relu = nn.ReLU()

    def forward(self, x, hss=None, return_hs=False):
        if hss is None:
            hss = [None, None]

        x, hss[0] = self.rec1(x, hss[0])

        n_batches, n_seq, _ = x.shape

        x = x.reshape(n_batches * n_seq, 1,
                      self.n_hidden_sqrt, self.n_hidden_sqrt)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2(x)
        x = x.view(n_batches, n_seq, -1)
        x = self.fc_out1(x)
        x = self.relu(x)
        x = self.fc_out2(x)

        if return_hs:
            return x, hss
        return x


class ConvWordTextModel2(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size=1, n_hidden_sqrt=24):
        super().__init__()
        if n_hidden_sqrt % 2 != 0:
            raise Exception("n_hidden_sqrt must be divisible by 2")

        self.embedding = nn.Embedding(vocab_size * context_size, embedding_dim)

        n_hidden = n_hidden_sqrt ** 2

        self.n_hidden_sqrt = n_hidden_sqrt
        self.fc1 = nn.Linear(context_size * embedding_dim, n_hidden)

        self.gru = nn.GRU(n_hidden, n_hidden, batch_first=True)

        self.lstm = nn.LSTM(n_hidden, n_hidden, batch_first=True)

        self.conv = nn.Conv2d(1, 16, 3, stride=1, padding=1)

        conv_out_size = n_hidden_sqrt

        self.max_pool = nn.MaxPool2d([2, 2], stride=2, padding=0)

        conv_out_size /= 2

        self.pool_out_size = int(16 * (conv_out_size ** 2))

        self.bil_fc = nn.Linear(self.pool_out_size + n_hidden, n_hidden)

        self.fc2 = nn.Linear(n_hidden, vocab_size)

        self.relu = nn.ReLU()

    def forward(self, x, hss=None, return_hs=False):
        if hss is None:
            hss = [None, None]

        n_batches, n_seq = x.shape[:2]
        
        x = self.embedding(x)

        x = x.view(n_batches, n_seq, -1)

        x = self.fc1(x)
        x = self.relu(x)

        x, hss[0] = self.gru(x, hss[0])

        x_conv, hss[1] = self.lstm(x, hss[1])

        

        x_conv = x_conv.reshape(n_batches * n_seq, 1, self.n_hidden_sqrt, self.n_hidden_sqrt)

        x_conv = self.conv(x_conv)
        x_conv = self.max_pool(x_conv)

        x_conv = x_conv.view(n_batches, n_seq, -1)

        x = torch.cat([x_conv, x], dim=2)

        x = self.bil_fc(x)
        x = self.relu(x)
        x = self.fc2(x)

        if return_hs:
            return x, hss
        return x

class ConvWordTextModel(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 context_size=1,
                 n_hidden_sqrt=32,
                 n_rec_layers=1,
                 ):
        super(ConvWordTextModel, self).__init__()

        if n_hidden_sqrt % 4 != 0:
            raise Exception("n_hidden_sqrt must be divisible by 4")

        self.embedding = nn.Embedding(vocab_size * context_size, embedding_dim)

        n_hidden = n_hidden_sqrt ** 2
        self.n_hidden_sqrt = n_hidden_sqrt

        self.fc1 = nn.Linear(embedding_dim, n_hidden)

        self.rec1 = nn.GRU(n_hidden, n_hidden, batch_first=True,
                           num_layers=n_rec_layers)

        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)

        conv_out_size = n_hidden_sqrt

        self.max_pool1 = nn.MaxPool2d([2, 2], stride=2, padding=0)

        conv_out_size /= 2

        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)

        self.max_pool2 = nn.MaxPool2d([2, 2], stride=2, padding=0)

        conv_out_size /= 2

        self.pool_out_size = int(32 * (conv_out_size ** 2))

        n = int(abs(self.pool_out_size - vocab_size) /
                2 + min(self.pool_out_size, vocab_size))

        self.hs_to_rec2 = nn.Linear(n_hidden, self.pool_out_size)

        self.rec2 = nn.GRU(self.pool_out_size, self.pool_out_size, batch_first=True)

        self.fc_out1 = nn.Linear(self.pool_out_size, vocab_size)

        self.relu = nn.ReLU()

    def forward(self, x, hss=None, return_hs=False):
        x = self.embedding(x)

        x = self.fc1(x)
        x = self.relu(x)
        x, hss = self.rec1(x, hss)
        n_batches, n_seq, _ = x.shape

        x = x.reshape(n_batches * n_seq, 1, self.n_hidden_sqrt, self.n_hidden_sqrt)

        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)

        x = x.view(n_batches, n_seq, -1)

        hss2 = self.hs_to_rec2(hss)

        x, _ = self.rec2(x, hss2)

        x = self.fc_out1(x)

        if return_hs:
            return x, hss
        return x

class SimpleWordTextModel(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 n_hidden=500,
                 n_rec_layers=1,
                 ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.rec1 = nn.GRU(embedding_dim, n_hidden, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.fc_to_out = nn.Linear(n_hidden, vocab_size)
        self.relu = nn.ReLU()


    def forward(self, x, hss=None, return_hs=False):
        x = self.embedding(x)
        x, hss = self.rec1(x, hss)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc_to_out(x)

        if return_hs:
            return x, hss
        return x
        