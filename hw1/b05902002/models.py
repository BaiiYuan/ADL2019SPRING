import torch
import torch.nn as nn
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"

class RNNbase(nn.Module):
    def __init__(self, input_size=300, classes=1, hidden_size=32, window_size=128,
                 drop_p=0.1, num_of_words=80000):
        super(RNNbase, self).__init__()

        self.word_embedding = torch.nn.Embedding(num_of_words, 300)
        self.dropout = nn.Dropout(drop_p)
        self.num_layers = 2

        self.lstm1_rec = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=self.num_layers, dropout=drop_p, bidirectional=True)

        self.dense_rec = nn.Linear(hidden_size*2*3, hidden_size)
        self.dense_rep = nn.Linear(hidden_size*2*3, hidden_size)

        self.weight = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, input_data_rec, input_data_rep):
        batch_size = input_data_rec.shape[0]

        x_rec = self.word_embedding(input_data_rec)
        x_rep = self.word_embedding(input_data_rep)

        rnn_output1_rec, (hn_1_rec, cn_1_rec) = self.lstm1_rec(x_rec)
        rnn_output1_rep, (hn_1_rep, cn_1_rep) = self.lstm1_rec(x_rep)

        last_rec = rnn_output1_rec[:,-1]
        mean_rec = rnn_output1_rec.mean(dim=1)
        max_rec = rnn_output1_rec.max(dim=1)[0]
        output_rec = torch.cat((last_rec, max_rec, mean_rec), dim=1)

        last_rep = rnn_output1_rep[:,-1]
        mean_rep = rnn_output1_rep.mean(dim=1)
        max_rep = rnn_output1_rep.max(dim=1)[0]
        output_rep = torch.cat((last_rep, max_rep, mean_rep), dim=1)

        output_rec = self.dense_rec(self.dropout(output_rec))
        output_rep = self.dense_rep(self.dropout(output_rep))


        # inner product batch-wise
        output_rec = self.weight(output_rec)
        pred = torch.bmm(output_rec.view(batch_size, 1, -1), output_rep.view(batch_size, -1, 1))
        pred = pred.squeeze()

        return pred


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
        self.lstm2_rec = nn.LSTM(self.hidden_size*8, self.hidden_size, batch_first=True, bidirectional=True) #, num_layers=self.num_layers, dropout=self.drop_p)

        self.dense_rec = nn.Linear(hidden_size*2*3, hidden_size)
        self.dense_rep = nn.Linear(hidden_size*2*3, hidden_size)

        self.weight1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, input_data_rec, input_data_rep):
        batch_size = input_data_rec.shape[0]

        x_rec = self.dropout(self.word_embedding(input_data_rec))
        x_rep = self.dropout(self.word_embedding(input_data_rep))

        lstm_output1_rec, _ = self.lstm1_rec(x_rec)
        lstm_output1_rep, _ = self.lstm1_rec(x_rep)

        raw_att2rep = F.softmax(torch.bmm(lstm_output1_rec, lstm_output1_rep.transpose(1,2)), dim=2)
        raw_att2rec = F.softmax(torch.bmm(lstm_output1_rep, lstm_output1_rec.transpose(1,2)), dim=2)

        ## Attention Weight
        attn_applied_rep = torch.bmm(raw_att2rec, lstm_output1_rec)
        attn_applied_rec = torch.bmm(raw_att2rep, lstm_output1_rep)

        attn_applied_rec = torch.cat((attn_applied_rec, lstm_output1_rec, attn_applied_rec*lstm_output1_rec, attn_applied_rec-lstm_output1_rec), dim=2)
        attn_applied_rep = torch.cat((attn_applied_rep, lstm_output1_rep, attn_applied_rep*lstm_output1_rep, attn_applied_rep-lstm_output1_rep), dim=2)

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

        return pred, (raw_att2rep, raw_att2rec)
