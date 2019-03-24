import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from IPython import embed

device = "cuda" if torch.cuda.is_available else "cpu"
class RNNbase(nn.Module):
    def __init__(self, input_size=300, classes=1, embedding_size=256, hidden_size=32, window_size=128, drop_p=0.3, num_of_words=80000):
        super(RNNbase, self).__init__()

        self.input_size = input_size
        self.classes = classes
        self.embedding_size = embedding_size
        self.word_embedding = torch.nn.Embedding(num_of_words, 300)
        self.hidden_size = hidden_size
        self.window_size = window_size

        self.dropout = nn.Dropout(drop_p)

        # self.embedd = nn.Sequential(
        #               nn.Linear(self.input_size, self.embedding_size),
        #               nn.ReLU()
        #               )

        self.gru1_rec = nn.GRU(self.input_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.gru1_rep = nn.GRU(self.input_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.gru2_rec = nn.GRU(self.hidden_size*2, self.hidden_size, batch_first=True, bidirectional=True)
        self.gru2_rep = nn.GRU(self.hidden_size*2, self.hidden_size, batch_first=True, bidirectional=True)
        # self.lstm1_rec = nn.LSTM(self.embedding_size, self.hidden_size, batch_first=True, bidirectional=True)
        # self.lstm1_rep = nn.LSTM(self.embedding_size, self.hidden_size, batch_first=True, bidirectional=True)


        self.weight = nn.Linear(self.hidden_size*2*2, self.hidden_size*2*2)

    def forward(self, input_data_rec, input_data_rep):
        batch_size = input_data_rec.shape[0]

        x_rec = self.word_embedding(input_data_rec)
        x_rep = self.word_embedding(input_data_rep)

        rnn_output1_rec, hn_1_rec = self.gru1_rec(x_rec) # hn1: (1, batch, hidden)
        rnn_output1_rep, hn_1_rep = self.gru1_rep(x_rep) # hn1: (1, batch, hidden)

        rnn_output1_rec = self.dropout(rnn_output1_rec)
        rnn_output1_rep = self.dropout(rnn_output1_rep)

        rnn_output2_rec, hn_2_rec = self.gru2_rec(rnn_output1_rec) # hn2: (1, batch, hidden)
        rnn_output2_rep, hn_2_rep = self.gru2_rep(rnn_output1_rep) # hn2: (1, batch, hidden)

        last_rec = rnn_output2_rec[:,-1]
        mean_rec = rnn_output2_rec.mean(dim=1)
        max_rec = rnn_output2_rec.max(dim=1)[0]
        output_rec = torch.cat((mean_rec, last_rec), dim=1)

        last_rep = rnn_output2_rep[:,-1]
        mean_rep = rnn_output2_rep.mean(dim=1)
        max_rep = rnn_output2_rep.max(dim=1)[0]
        output_rep = torch.cat((mean_rep, last_rep), dim=1)

        # output_rec = rnn_output2_rec[:,-1]
        # output_rep = rnn_output2_rep[:,-1]

        # inner product batch-wise
        output_rec = self.dropout(self.weight(output_rec))
        output_rec = F.relu(output_rec)
        pred = torch.bmm(output_rec.view(batch_size, 1, -1), output_rep.view(batch_size, -1, 1))
        pred = pred.squeeze()

        # embed()

        return pred

class RNNatt(nn.Module):
    def __init__(self, input_size=300, classes=1, embedding_size=512, hidden_size=256, rec_len=112, rep_len=16, window_size=128, drop_p=0.3, num_of_words=80000):
        super(RNNatt, self).__init__()

        self.input_size = input_size
        self.classes = classes
        self.embedding_size = embedding_size
        self.word_embedding = torch.nn.Embedding(num_of_words, 300)
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.dropout = nn.Dropout(drop_p)

        self.gru1_rec = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.gru1_rep = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.attn1_rec = nn.Linear(self.hidden_size*2, rec_len)
        self.attn1_rep = nn.Linear(self.hidden_size*2, rep_len)


        self.gru2_rec = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.gru2_rep = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True, bidirectional=True)
        # self.attn2_rec = nn.Linear(self.hidden_size*2, rec_len)
        # self.attn2_rep = nn.Linear(self.hidden_size*2, rep_len)

        # self.gru3_rec = nn.GRU(self.hidden_size*2, self.hidden_size, batch_first=True, bidirectional=True)
        # self.gru3_rep = nn.GRU(self.hidden_size*2, self.hidden_size, batch_first=True, bidirectional=True)


        self.weight1 = nn.Linear(self.hidden_size*2*3, self.hidden_size*2*3)
        # self.weight2 = nn.Linear(self.hidden_size*2, self.hidden_size*2)


    def forward(self, input_data_rec, input_data_rep):
        batch_size = input_data_rec.shape[0]

        x_rec = self.word_embedding(input_data_rec)
        x_rep = self.word_embedding(input_data_rep)

        rnn_output1_rec, hn_1_rec = self.gru1_rec(x_rec) # hn1: (1, batch, hidden)
        rnn_output1_rep, hn_1_rep = self.gru1_rep(x_rep) # hn1: (1, batch, hidden)
        rnn_output1_rec = self.dropout(rnn_output1_rec)
        rnn_output1_rep = self.dropout(rnn_output1_rep)

        ## Attention Weight
        attn_weight_rec = F.softmax(self.attn1_rec(rnn_output1_rec), dim=2)
        attn_weight_rep = F.softmax(self.attn1_rep(rnn_output1_rep), dim=2)

        attn_applied_rec = torch.bmm(attn_weight_rec, rnn_output1_rec)
        attn_applied_rep = torch.bmm(attn_weight_rep, rnn_output1_rep)

        rnn_output2_rec, hn_2_rec = self.gru2_rec(attn_applied_rec)
        rnn_output2_rep, hn_2_rep = self.gru2_rep(attn_applied_rep)
        rnn_output2_rec = self.dropout(rnn_output2_rec)
        rnn_output2_rep = self.dropout(rnn_output2_rep)

        # attn_weight_rec = F.softmax(self.attn2_rec(rnn_output2_rec), dim=2)
        # attn_weight_rep = F.softmax(self.attn2_rep(rnn_output2_rep), dim=2)
        # attn_applied_rec = torch.bmm(attn_weight_rec, rnn_output2_rec)
        # attn_applied_rep = torch.bmm(attn_weight_rep, rnn_output2_rep)

        # rnn_output3_rec, hn_3_rec = self.gru3_rec(attn_applied_rec)
        # rnn_output3_rep, hn_3_rep = self.gru3_rep(attn_applied_rep)


        last_rec = rnn_output2_rec[:,-1]
        mean_rec = rnn_output2_rec.mean(dim=1)
        max_rec = rnn_output2_rec.max(dim=1)[0]
        output_rec = torch.cat((mean_rec, max_rec, last_rec), dim=1)

        last_rep = rnn_output2_rep[:,-1]
        mean_rep = rnn_output2_rep.mean(dim=1)
        max_rep = rnn_output2_rep.max(dim=1)[0]
        output_rep = torch.cat((mean_rep, max_rep, last_rep), dim=1)

        # inner product batch-wise
        output_rec = self.dropout(self.weight1(output_rec))
        # output_rec = F.relu(output_rec)
        # output_rec = self.dropout(self.weight2(output_rec))
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