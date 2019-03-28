import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from IPython import embed

device = "cuda" if torch.cuda.is_available else "cpu"
class RNNbase(nn.Module):
    def __init__(self, input_size=300, classes=1, hidden_size=32, window_size=128, drop_p=0.1, num_of_words=80000):
        super(RNNbase, self).__init__()

        self.input_size = input_size
        self.classes = classes
        self.drop_p = drop_p
        self.word_embedding = torch.nn.Embedding(num_of_words, 300)
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.dropout = nn.Dropout(drop_p)
        self.num_layers = 2

        self.lstm1_rec = nn.LSTM(self.input_size, self.hidden_size, batch_first=True, num_layers=self.num_layers, dropout=self.drop_p, bidirectional=True)
        self.lstm1_rep = nn.LSTM(self.input_size, self.hidden_size, batch_first=True, num_layers=self.num_layers, dropout=self.drop_p, bidirectional=True)

        # self.dense_rec = nn.Linear(self.hidden_size*2*2, self.hidden_size*2)
        # self.dense_rep = nn.Linear(self.hidden_size*2*2, self.hidden_size*2)

        self.weight = nn.Linear(self.hidden_size*2*2, self.hidden_size*2*2)

    def forward(self, input_data_rec, input_data_rep):
        batch_size = input_data_rec.shape[0]

        x_rec = self.word_embedding(input_data_rec)
        x_rep = self.word_embedding(input_data_rep)

        rnn_output1_rec, (hn_1_rec, cn_1_rec) = self.lstm1_rec(x_rec)
        rnn_output1_rep, (hn_1_rep, cn_1_rep) = self.lstm1_rep(x_rep)

        # rnn_output1_rec = rnn_output1_rec.view(-1, batch_size, 2, self.hidden_size)
        # hn_1_rec = hn_1_rec.view(self.num_layers, 2, batch_size, self.hidden_size)
        # cn_1_rec = hn_1_rec.view(self.num_layers, 2, batch_size, self.hidden_size)

        last_rec = rnn_output1_rec[:,-1]
        mean_rec = rnn_output1_rec.mean(dim=1)
        max_rec = rnn_output1_rec.max(dim=1)[0]
        output_rec = torch.cat((max_rec, mean_rec), dim=1)

        last_rep = rnn_output1_rep[:,-1]
        mean_rep = rnn_output1_rep.mean(dim=1)
        max_rep = rnn_output1_rep.max(dim=1)[0]
        output_rep = torch.cat((max_rep, mean_rep), dim=1)

        # output_rec = self.dense_rec(self.dropout(output_rec))
        # output_rep = self.dense_rep(self.dropout(output_rep))


        # inner product batch-wise
        output_rec = self.weight(self.dropout(output_rec))
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
        self.lstm1_rep = nn.LSTM(self.input_size, self.hidden_size, batch_first=True, bidirectional=True)

        self.attn1_rec = nn.Linear(self.hidden_size*2, rec_len)
        self.attn1_rep = nn.Linear(self.hidden_size*2, rep_len)

        self.lstm2_rec = nn.LSTM(self.hidden_size*2+self.input_size, self.hidden_size, batch_first=True, bidirectional=True) #, num_layers=self.num_layers, dropout=self.drop_p)
        self.lstm2_rep = nn.LSTM(self.hidden_size*2+self.input_size, self.hidden_size, batch_first=True, bidirectional=True) #, num_layers=self.num_layers, dropout=self.drop_p)

        self.weight1 = nn.Linear(self.hidden_size*2*2, self.hidden_size*2*2)

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
        output_rec = torch.cat((mean_rec, max_rec), dim=1)

        last_rep = rnn_output_rep[:,-1]
        mean_rep = rnn_output_rep.mean(dim=1)
        max_rep = rnn_output_rep.max(dim=1)[0]
        output_rep = torch.cat((mean_rep, max_rep), dim=1)

        # inner product batch-wise
        output_rec = self.weight1(output_rec)
        pred = torch.bmm(output_rec.view(batch_size, 1, -1), output_rep.view(batch_size, -1, 1))
        pred = pred.squeeze()

        return pred




class Encoder(nn.Module):
    def __init__(self, input_size=300, hidden_size=256, nlayers=1, dropout=0.,
                 bidirectional=True, rnn_type='GRU'):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.lstm_rec = nn.LSTM(self.input_size, self.hidden_size, nlayers,
                                 dropout=dropout, bidirectional=bidirectional)

    def forward(self, x_rec, x_rep, hidden=None):
        batch_size = x_rec.shape[0]

        return self.lstm_rec(x_rec, hidden), self.lstm_rec(x_rep, hidden)


class Attention(nn.Module):
  def __init__(self, query_dim, key_dim, value_dim):
    super(Attention, self).__init__()
    self.dim = query_dim
    self.scale = 1. / math.sqrt(query_dim)

  def forward(self, query, keys, values):
    # Query = [BxTxQ] 100/75/128
    # Keys = [BxTxK] 100/75/128
    # Values = [BxTxV] 100/75/256
    # Outputs = a:[BxTxT], lin_comb:[BxTxV]

    keys = keys.transpose(1,2)
    energy = torch.bmm(query, keys)
    energy = F.softmax(energy.mul_(self.scale), dim=2) # scale, normalize

    linear_combination = torch.bmm(energy, values).squeeze(1)
    return energy, linear_combination

class Classifier(nn.Module):
  def __init__(self, encoder1, encoder2, attention, hidden_size, num_classes, num_of_words=80000,
               rec_len=112, rep_len=16, input_size=300, drop_p=0.3):
    super(Classifier, self).__init__()
    self.encoder1 = encoder1
    self.encoder2 = encoder2
    self.attention = attention

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.rec_len = rec_len
    self.word_embedding = torch.nn.Embedding(num_of_words, 300)
    self.dropout = nn.Dropout(drop_p)

    self.q_ = nn.Linear(self.input_size, self.hidden_size)
    self.k_ = nn.Linear(self.input_size, self.hidden_size)

    self.weight1 = nn.Linear(self.hidden_size*2*2, self.hidden_size*2*2)

    size = 0
    for p in self.parameters():
        size += p.nelement()
    print('Total param size: {}'.format(size))


  def forward(self, input_data_rec, input_data_rep):
    x_rec = self.word_embedding(input_data_rec)
    x_rep = self.word_embedding(input_data_rep)
    batch_size =  x_rec.shape[0]

    (rec_outputs, rec_hidden), (rep_outputs, rep_hidden) = self.encoder1(x_rec, x_rep)
    # rec_hidden = rec_hidden[1] # take the cell state
    # rep_hidden = rep_hidden[1] # take the cell state

    # rec_hidden = torch.cat([rec_hidden[-1], rec_hidden[-2]], dim=1)
    # rep_hidden = torch.cat([rep_hidden[-1], rep_hidden[-2]], dim=1)

    outputs = torch.cat((x_rec, x_rep), dim=1)
    V = torch.cat((rec_outputs, rep_outputs), dim=1)

    Q = self.q_(self.dropout(outputs))
    K = self.k_(self.dropout(outputs))

    # max across T?
    # Other options (work worse on a few tests):
    # linear_combination, _ = torch.max(outputs, 0)
    # linear_combination = torch.mean(outputs, 0)

    energy, linear_combination = self.attention(Q, K, V)

    comb_rec = linear_combination[:, :self.rec_len]
    comb_rep = linear_combination[:, self.rec_len:]

    (rnn_output_rec, _), (rnn_output_rep, _) = self.encoder2(comb_rec, comb_rep)

    last_rec = rnn_output_rec[:,-1]
    mean_rec = rnn_output_rec.mean(dim=1)
    max_rec = rnn_output_rec.max(dim=1)[0]
    output_rec = torch.cat((mean_rec, max_rec), dim=1)

    last_rep = rnn_output_rep[:,-1]
    mean_rep = rnn_output_rep.mean(dim=1)
    max_rep = rnn_output_rep.max(dim=1)[0]
    output_rep = torch.cat((mean_rep, max_rep), dim=1)

    # inner product batch-wise
    output_rec = self.weight1(self.dropout(output_rec))
    pred = torch.bmm(output_rec.view(batch_size, 1, -1), output_rep.view(batch_size, -1, 1))
    pred = pred.squeeze()

    return pred, energy


if __name__ == "__main__":
    model = RNNatt()
    seq_in1 = torch.randn(24, 112).long()**2
    seq_in2 = torch.randn(24, 16).long()**2
    # embed()
    y = model(seq_in1, seq_in2)
    print("seq_in: {}, {}, y: {}".format(seq_in1.size(), seq_in2.size(), y.size()))