import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from IPython import embed

device = "cuda" if torch.cuda.is_available else "cpu"
class RNNbase(nn.Module):
    def __init__(self, input_size=300, classes=1, embedding_size=512, hidden_size=256, window_size=128, drop_p=0.2, num_of_words=80000):
        super(RNNbase, self).__init__()

        self.input_size = input_size
        self.classes = classes
        self.embedding_size = embedding_size
        self.word_embedding = torch.nn.Embedding(num_of_words, 300)
        self.hidden_size = hidden_size
        self.window_size = window_size

        self.dropout = nn.Dropout(drop_p)

        self.embedd = nn.Sequential(
                      nn.Linear(self.input_size, self.embedding_size),
                      nn.ReLU() # Can not add Dropout
                      )

        self.gru1_rec = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.gru1_rep = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True, bidirectional=True)
        # for i in range(4):
        #     self.gru1.all_weights[0][i] = nn.init.xavier_normal_(self.gru1.all_weights[0][i])

        # self.gru2 = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True)
        self.classify = nn.Sequential(
                        # nn.Dropout(0.2),
                        nn.Linear(self.hidden_size*2, 32),
                        nn.ReLU(),
                        # nn.Dropout(0.2),
                        nn.Linear(32, self.classes),
                        # nn.Sigmoid()
                        )
        self.weight = nn.Linear(self.hidden_size*2, self.hidden_size*2)

    def forward(self, input_data_rec, input_data_rep):
        batch_size = input_data_rec.shape[0]

        x_rec = self.word_embedding(input_data_rec)
        x_rep = self.word_embedding(input_data_rep)

        x_rec = F.relu(self.embedd(x_rec))
        x_rep = F.relu(self.embedd(x_rep))

        rnn_output1_rec, hn_1_rec = self.gru1_rec(x_rec) # hn1: (1, batch, hidden)
        rnn_output1_rep, hn_1_rep = self.gru1_rep(x_rep) # hn1: (1, batch, hidden)

        output_rec = rnn_output1_rec.mean(dim=1)
        output_rep = rnn_output1_rec.mean(dim=1)

        # concat = torch.cat((output_rec, output_rep), dim=1)
        # pred = self.classify(concat).squeeze()

        # inner product batch-wise
        output_rec = self.weight(output_rec)
        output_rec = F.relu(output_rec)
        pred = torch.bmm(output_rec.view(batch_size, 1, -1), output_rep.view(batch_size, -1, 1))
        pred = pred.squeeze()


        # rnn_output2, hn_2 = self.gru2(rnn_output1) # hn2: (1, batch, hidden)
        # pred = self.classify(rnn_output1[:, 0])
        return pred

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
    seq_in1 = torch.randn(24, 112, 300)
    seq_in2 = torch.randn(24, 16, 300)

    y = model(seq_in1, seq_in2)
    print("seq_in: {}, {}, y: {}".format(seq_in1.size(), seq_in2.size(), y.size()))