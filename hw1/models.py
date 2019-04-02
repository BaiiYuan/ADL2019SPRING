import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from IPython import embed

device = "cuda" if torch.cuda.is_available else "cpu"

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
        size = 0
        for p in self.parameters():
            size += p.nelement()
        print('Total param size: {}'.format(size))

    def forward(self, input_data_rec, input_data_rep):
        batch_size = input_data_rec.shape[0]

        x_rec = self.word_embedding(input_data_rec)
        x_rep = self.word_embedding(input_data_rep)

        rnn_output1_rec, (hn_1_rec, cn_1_rec) = self.lstm1_rec(x_rec)
        rnn_output1_rep, (hn_1_rep, cn_1_rep) = self.lstm1_rec(x_rep)

        # rnn_output1_rec = rnn_output1_rec.view(-1, batch_size, 2, self.hidden_size)
        # hn_1_rec = hn_1_rec.view(self.num_layers, 2, batch_size, self.hidden_size)
        # cn_1_rec = hn_1_rec.view(self.num_layers, 2, batch_size, self.hidden_size)

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
        output_rec = self.weight(self.dropout(output_rec))
        pred = torch.bmm(output_rec.view(batch_size, 1, -1), output_rep.view(batch_size, -1, 1))
        pred = pred.squeeze()

        return pred


class Encoder(nn.Module):
    def __init__(self, input_size=300, hidden_size=256, nlayers=1, dropout=0.,
                 bidirectional=True, rnn_type='GRU'):
        super(Encoder, self).__init__()
        self.lstm_rec = nn.LSTM(input_size, hidden_size, nlayers,
                                 dropout=dropout, bidirectional=bidirectional)

    def forward(self, x_rec, x_rep, hidden=None):
        batch_size = x_rec.shape[0]
        return self.lstm_rec(x_rec, hidden), self.lstm_rec(x_rep, hidden)

class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super(Attention, self).__init__()
        self.dim = int(query_dim)
        self.scale = 1./ math.sqrt(query_dim)

    def forward(self, query, keys, values):
        keys = keys.transpose(1,2)
        energy = torch.bmm(query, keys)
        energy = F.softmax(energy.mul_(self.scale), dim=2) # scale, normalize

        linear_combination = torch.bmm(energy, values).squeeze(1)
        return energy, linear_combination

class Classifier(nn.Module):
    def __init__(self, encoder1, encoder2, attention, hidden_size, num_of_words=80000,
                 rec_len=112, rep_len=16, input_size=300, drop_p=0.0):
        super(Classifier, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.attention = attention
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rec_len = rec_len
        self.word_embedding = torch.nn.Embedding(num_of_words, 300)
        self.dropout = nn.Dropout(drop_p)

        self.q_ = nn.Linear(self.input_size, self.attention.dim, bias=False)
        self.k_ = nn.Linear(self.input_size, self.attention.dim, bias=False)

        self.weight1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.dense_rec = nn.Linear(hidden_size*2*3, hidden_size)
        self.dense_rep = nn.Linear(hidden_size*2*3, hidden_size)

        size = 0
        for p in self.parameters():
            size += p.nelement()
        print('Total param size: {}'.format(size))


    def forward(self, input_data_rec, input_data_rep):
        x_rec = self.word_embedding(input_data_rec)
        x_rep = self.word_embedding(input_data_rep)
        batch_size =  x_rec.shape[0]

        (rec_outputs, rec_hidden), (rep_outputs, rep_hidden) = self.encoder1(x_rec, x_rep)

        outputs = torch.cat((x_rec, x_rep), dim=1)
        V = torch.cat((rec_outputs, rep_outputs), dim=1)

        Q = self.q_(self.dropout(outputs))
        K = self.k_(self.dropout(outputs))

        energy, linear_combination = self.attention(Q, K, V)

        comb_rec = linear_combination[:, :self.rec_len]
        comb_rep = linear_combination[:, self.rec_len:]

        comb_rec = torch.cat((rec_outputs, comb_rec, rec_outputs-comb_rec, rec_outputs*comb_rec), dim=2)
        comb_rep = torch.cat((rep_outputs, comb_rep, rep_outputs-comb_rep, rep_outputs*comb_rep), dim=2)
        (rnn_output_rec, _), (rnn_output_rep, _) = self.encoder2(comb_rec, comb_rep)

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
        output_rec = self.weight1(self.dropout(output_rec))
        pred = torch.bmm(output_rec.view(batch_size, 1, -1), output_rep.view(batch_size, -1, 1))
        pred = pred.squeeze()

        return pred, energy



class BiDAF(nn.Module):
    def __init__(self, input_size=300, classes=1, hidden_size=256, window_size=128,
                 drop_p=0.2, num_of_words=80000):
        super(BiDAF, self).__init__()

        self.drop_p = drop_p
        self.word_embedding = torch.nn.Embedding(num_of_words, 300)
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(drop_p)
        self.num_layers = 2

        self.lstm1_rec = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)

        # self.W = nn.Linear(hidden_size*6, 1, bias=False)
        self.modeling_layer = nn.LSTM(hidden_size*8, hidden_size, num_layers=2, bidirectional=True, dropout=drop_p, batch_first=True)

        self.denseOut = nn.Sequential(
                        nn.Linear(self.hidden_size*6, self.hidden_size),
                        nn.ReLU(),
                        nn.Linear(self.hidden_size, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1)
                        )

        size = 0
        for p in self.parameters():
            size += p.nelement()
        print('Total param size: {}'.format(size-17188200))

    def forward(self, input_data_rec, input_data_rep):
        batch_size = input_data_rec.shape[0]
        T = input_data_rec.shape[1]
        J = input_data_rep.shape[1]

        x_rec = self.word_embedding(input_data_rec)
        x_rep = self.word_embedding(input_data_rep)

        embd_context, (hn_1_rec, cn_1_rec) = self.lstm1_rec(x_rec)
        embd_query, (hn_1_rep, cn_1_rep) = self.lstm1_rec(x_rep)

        # shape = (batch_size, T, J, -1)
        # embd_context_ex = embd_context.unsqueeze(2)     # (N, T, 1, 2d)
        # embd_context_ex = embd_context_ex.expand(shape) # (N, T, J, 2d)
        # embd_query_ex = embd_query.unsqueeze(1)         # (N, 1, J, 2d)
        # embd_query_ex = embd_query_ex.expand(shape)     # (N, T, J, 2d)
        # a_elmwise_mul_b = torch.mul(embd_context_ex, embd_query_ex) # (N, T, J, 2d)
        # cat_data = torch.cat((embd_context_ex, embd_query_ex, a_elmwise_mul_b), 3) # (N, T, J, 6d), [h;u;h◦u]
        # S = self.W(cat_data).view(batch_size, T, J)

        S = torch.bmm(embd_context, embd_query.transpose(1, 2))

        c2q = torch.bmm(F.softmax(S, dim=-1), embd_query)
        b = F.softmax(torch.max(S, 2)[0], dim=-1)
        q2c = torch.bmm(b.unsqueeze(1), embd_context)
        q2c = q2c.expand(-1, T, -1)
        # q2c = q2c.repeat(1, T, 1)

        G = torch.cat((embd_context, c2q, embd_context.mul(c2q), embd_context.mul(q2c)), 2)
        M, _h = self.modeling_layer(G)

        last_M = M[:,-1]
        mean_M = M.mean(dim=1)
        max_M = M.max(dim=1)[0]
        output_M = torch.cat((last_M, max_M, mean_M), dim=1)

        pred = self.denseOut(output_M)

        return pred.squeeze()


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

        # self.attn1_rec = nn.Linear(self.hidden_size*2, rec_len)
        # self.attn1_rep = nn.Linear(self.hidden_size*2, rep_len)

        self.lstm2_rec = nn.LSTM(self.hidden_size*8, self.hidden_size, batch_first=True, bidirectional=True) #, num_layers=self.num_layers, dropout=self.drop_p)

        self.dense_rec = nn.Linear(hidden_size*2*3, hidden_size)
        self.dense_rep = nn.Linear(hidden_size*2*3, hidden_size)

        self.weight1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, input_data_rec, input_data_rep):
        batch_size = input_data_rec.shape[0]

        x_rec = self.dropout(self.word_embedding(input_data_rec))
        x_rep = self.dropout(self.word_embedding(input_data_rep))

        lstm_output1_rec, (hn_1_rec, cn_1_rec) = self.lstm1_rec(x_rec)
        lstm_output1_rep, (hn_1_rep, cn_1_rep) = self.lstm1_rec(x_rep)

        raw_att2rep = F.softmax(torch.bmm(lstm_output1_rec, lstm_output1_rep.transpose(1,2)), dim=2)
        raw_att2rec = F.softmax(torch.bmm(lstm_output1_rep, lstm_output1_rec.transpose(1,2)), dim=2)

        ## Attention Weight
        # attn_weight_rec = F.softmax(self.attn1_rec(lstm_output1_rec), dim=2)
        # attn_weight_rep = F.softmax(self.attn1_rep(lstm_output1_rep), dim=2)

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

        return pred



if __name__ == "__main__":
    model = RNNatt()
    seq_in1 = torch.randn(24, 112).long()**2
    seq_in2 = torch.randn(24, 16).long()**2
    # embed()
    y = model(seq_in1, seq_in2)
    print("seq_in: {}, {}, y: {}".format(seq_in1.size(), seq_in2.size(), y.size()))