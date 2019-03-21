import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from IPython import embed

device = "cuda" if torch.cuda.is_available else "cpu"
class RNNbase(nn.Module):
    def __init__(self, input_size=300, classes=1, embedding_size=256, hidden_size=32, window_size=128, drop_p=0.2, num_of_words=80000):
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
                      nn.ReLU()
                      )

        self.gru1_rec = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.gru1_rep = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.gru2_rec = nn.GRU(self.hidden_size*2, self.hidden_size, batch_first=True, bidirectional=True)
        self.gru2_rep = nn.GRU(self.hidden_size*2, self.hidden_size, batch_first=True, bidirectional=True)
        # self.lstm1_rec = nn.LSTM(self.embedding_size, self.hidden_size, batch_first=True, bidirectional=True)
        # self.lstm1_rep = nn.LSTM(self.embedding_size, self.hidden_size, batch_first=True, bidirectional=True)

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

        x_rec = self.dropout(F.relu(self.embedd(x_rec)))
        x_rep = self.dropout(F.relu(self.embedd(x_rep)))

        rnn_output1_rec, hn_1_rec = self.gru1_rec(x_rec) # hn1: (1, batch, hidden)
        rnn_output1_rep, hn_1_rep = self.gru1_rep(x_rep) # hn1: (1, batch, hidden)

        rnn_output1_rec = self.dropout(rnn_output1_rec)
        rnn_output1_rep = self.dropout(rnn_output1_rep)

        rnn_output2_rec, hn_2_rec = self.gru2_rec(rnn_output1_rec) # hn2: (1, batch, hidden)
        rnn_output2_rep, hn_2_rep = self.gru2_rep(rnn_output1_rep) # hn2: (1, batch, hidden)

        output_rec = rnn_output1_rec[:,-1]
        output_rep = rnn_output1_rep[:,-1]

        # inner product batch-wise
        output_rec = self.dropout(self.weight(output_rec))
        output_rec = F.relu(output_rec)
        pred = torch.bmm(output_rec.view(batch_size, 1, -1), output_rep.view(batch_size, -1, 1))
        pred = pred.squeeze()

        # concat = torch.cat((output_rec, output_rep), dim=1)
        # pred = self.classify(concat).squeeze()

        embed()

        return pred

class RNNatt(nn.Module):
    def __init__(self, input_size=300, classes=1, embedding_size=256, hidden_size=32, window_size=128, drop_p=0.2, num_of_words=80000):
        super(RNNatt, self).__init__()

        self.input_size = input_size
        self.classes = classes
        self.embedding_size = embedding_size
        self.word_embedding = torch.nn.Embedding(num_of_words, 300)
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.dropout = nn.Dropout(drop_p)
        self.embedd = nn.Sequential(
                      nn.Linear(self.input_size, self.embedding_size),
                      nn.ReLU()
                      )

        self.gru1_rec = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.gru1_rep = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True, bidirectional=True)

        self.attn_rec = nn.Linear(self.hidden_size*2, 112*112)
        self.attn_rep = nn.Linear(self.hidden_size*2, 16*16)

        self.gru2_rec = nn.GRU(self.hidden_size*2, self.hidden_size, batch_first=True, bidirectional=True)
        self.gru2_rep = nn.GRU(self.hidden_size*2, self.hidden_size, batch_first=True, bidirectional=True)

        self.weight = nn.Linear(self.hidden_size*2, self.hidden_size*2)

    def forward(self, input_data_rec, input_data_rep):
        batch_size = input_data_rec.shape[0]

        x_rec = self.word_embedding(input_data_rec)
        x_rep = self.word_embedding(input_data_rep)

        x_rec = self.dropout(F.relu(self.embedd(x_rec)))
        x_rep = self.dropout(F.relu(self.embedd(x_rep)))

        rnn_output1_rec, hn_1_rec = self.gru1_rec(x_rec) # hn1: (1, batch, hidden)
        rnn_output1_rep, hn_1_rep = self.gru1_rep(x_rep) # hn1: (1, batch, hidden)
        rnn_output1_rec = self.dropout(rnn_output1_rec)
        rnn_output1_rep = self.dropout(rnn_output1_rep)

        attn_weight_rec = self.attn_rec(hn_1_rec.view(batch_size, -1))
        attn_weight_rep = self.attn_rep(hn_1_rep.view(batch_size, -1))

        ## Attention PART
        attn_weight_rec = F.softmax(attn_weight_rec.view(batch_size, 112, 112), dim=2)
        attn_weight_rep = F.softmax(attn_weight_rep.view(batch_size, 16, 16), dim=2)
        attn_applied_rec = torch.bmm(attn_weight_rec, rnn_output1_rec)
        attn_applied_rep = torch.bmm(attn_weight_rep, rnn_output1_rep)

        rnn_output2_rec, hn_2_rec = self.gru2_rec(attn_applied_rec) # hn2: (1, batch, hidden)
        rnn_output2_rep, hn_2_rep = self.gru2_rep(attn_applied_rep) # hn2: (1, batch, hidden)

        output_rec = rnn_output1_rec[:,-1]
        output_rep = rnn_output1_rep[:,-1]

        # inner product batch-wise
        output_rec = self.dropout(self.weight(output_rec))
        output_rec = F.relu(output_rec)
        pred = torch.bmm(output_rec.view(batch_size, 1, -1), output_rep.view(batch_size, -1, 1))
        pred = pred.squeeze()

        return pred


if __name__ == "__main__":
    model = RNNatt()
    seq_in1 = torch.randn(24, 112).long()**2
    seq_in2 = torch.randn(24, 16).long()**2
    # embed()
    y = model(seq_in1, seq_in2)
    print("seq_in: {}, {}, y: {}".format(seq_in1.size(), seq_in2.size(), y.size()))