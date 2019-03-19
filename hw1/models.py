import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from IPython import embed

device = "cuda" if torch.cuda.is_available else "cpu"
class RNNbase(nn.Module):
    def __init__(self, input_size=300, classes=1, embedding_size=512, hidden_size=256, window_size=80, drop_p=0.2):
        super(RNNbase, self).__init__()

        self.input_size = input_size
        self.classes = classes
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        self.dropout = nn.Dropout(drop_p)

        self.embedd = nn.Sequential(
                      nn.Linear(self.input_size, self.embedding_size),
                      nn.ReLU() # Can not add Dropout
                      )

        self.gru1 = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True)
        # self.gru2 = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True)
        self.classify = nn.Sequential(
                        nn.Dropout(0.2),
                        nn.Linear(self.hidden_size, 32),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(32, self.classes),
                        # nn.Sigmoid()
                        )

    def forward(self, input_data):
        x = self.embedd(self.dropout(input_data))
        rnn_output1, hn_1 = self.gru1(x) # hn1: (1, batch, hidden)
        # rnn_output2, hn_2 = self.gru2(rnn_output1) # hn2: (1, batch, hidden)
        pred = self.classify(rnn_output1[:, 0])
        return pred.squeeze(1)

class RNNatt(nn.Module):
    def __init__(self, input_size=39, classes=5, embedding_size=32, hidden_size=128, window_size=128, drop_p=0.5):
        super(RNNatt, self).__init__()

        self.input_size = input_size
        self.classes = classes
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        self.dropout = nn.Dropout(drop_p)

        self.embedd = nn.Linear(self.input_size, self.embedding_size)

        self.rnn = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True)
        # self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, batch_first=True)
        self.attn = nn.Linear(self.hidden_size, self.window_size)

        self.classify = nn.Sequential(
                        nn.Linear(self.hidden_size, self.classes),
                        nn.LogSoftmax(dim=1)
                        )

    def forward(self, input_data):
        x = self.embedd(input_data+1e-2*torch.rand_like(input_data))
        x = F.relu(x)
        # embed()
        # x = x + 1e-2*torch.rand_like(x) # noise
        x = self.dropout(x)
        rnn_output, h_n = self.rnn(x) # h_n: (1, batch, hidden)
        # rnn_output, h_n = self.lstm(x);h_n, c_n = h_n
        att = self.attn(h_n[-1])
        attn_weights = F.softmax(att, dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), rnn_output)
        # attn_applied = attn_applied + 1e-2*torch.rand_like(attn_applied) # noise
        pred = self.classify(attn_applied.squeeze(1))

        return pred

if __name__ == "__main__":
    model = RNNbase()
    seq_in = torch.randn(24, 200, 300)

    y = model(seq_in)
    print("seq_in: {}, y: {}".format(seq_in.size(), y.size()))