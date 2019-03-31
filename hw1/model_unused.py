import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from IPython import embed

class RNNatt(nn.Module):
    def __init__(self, input_size=300, classes=1, hidden_size=256, rec_len=112, rep_len=16, window_size=128, drop_p=0.3, num_of_words=80000):
        super(RNNatt, self).__init__()

        self.input_size = input_size
        self.classes = classes
        self.drop_p = drop_p
        self.word_embedding = torch.nn.Embedding(num_of_words, 300)
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.dropout = nn.Dropout(drop_p)
        self.num_layers = 2

        self.lstm1_rec = nn.LSTM(self.input_size, self.hidden_size, batch_first=True, bidirectional=True)

        self.attn1_rec = nn.Linear(self.hidden_size*2, rec_len)
        self.attn1_rep = nn.Linear(self.hidden_size*2, rep_len)

        self.lstm2_rec = nn.LSTM(self.hidden_size*2+self.input_size, self.hidden_size, batch_first=True, bidirectional=True) #, num_layers=self.num_layers, dropout=self.drop_p)

        self.dense_rec = nn.Linear(hidden_size*2*3, hidden_size)
        self.dense_rep = nn.Linear(hidden_size*2*3, hidden_size)

        self.weight1 = nn.Linear(self.hidden_size*2*2, self.hidden_size*2*2, bias=False)

    def forward(self, input_data_rec, input_data_rep):
        batch_size = input_data_rec.shape[0]

        x_rec = self.dropout(self.word_embedding(input_data_rec))
        x_rep = self.dropout(self.word_embedding(input_data_rep))

        lstm_output1_rec, (hn_1_rec, cn_1_rec) = self.lstm1_rec(x_rec)
        lstm_output1_rep, (hn_1_rep, cn_1_rep) = self.lstm1_rec(x_rep)

        ## Attention Weight
        attn_weight_rec = F.softmax(self.attn1_rec(lstm_output1_rec), dim=2)
        attn_weight_rep = F.softmax(self.attn1_rep(lstm_output1_rep), dim=2)

        attn_applied_rec = torch.bmm(attn_weight_rec, lstm_output1_rec)
        attn_applied_rep = torch.bmm(attn_weight_rep, lstm_output1_rep)

        attn_applied_rec = torch.cat((attn_applied_rec, x_rec), dim=2)
        attn_applied_rep = torch.cat((attn_applied_rep, x_rep), dim=2)

        rnn_output_rec, _ = self.lstm2_rec(attn_applied_rec)
        rnn_output_rep, _ = self.lstm2_rec(attn_applied_rep)

        last_rec = rnn_output_rec[:,-1]
        mean_rec = rnn_output_rec.mean(dim=1)
        max_rec = rnn_output_rec.max(dim=1)[0]
        output_rec = torch.cat((last_rec, mean_rec, max_rec), dim=1)

        last_rep = rnn_output_rep[:,-1]
        mean_rep = rnn_output_rep.mean(dim=1)
        max_rep = rnn_output_rep.max(dim=1)[0]
        output_rep = torch.cat((last_rep, mean_rep, max_rep), dim=1)

        output_rec = self.dense_rec(self.dropout(output_rec))
        output_rep = self.dense_rep(self.dropout(output_rep))

        # inner product batch-wise
        output_rec = self.weight1(output_rec)
        pred = torch.bmm(output_rec.view(batch_size, 1, -1), output_rep.view(batch_size, -1, 1))
        pred = pred.squeeze()

        return pred

