import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

class elmo_model(nn.Module):
    def __init__(self, input_size=300, classes=1, hidden_size=32, window_size=128,
                 drop_p=0.1, num_of_words=80000):
        super(elmo_model, self).__init__()

        self.word_embedding = torch.nn.Embedding(num_of_words, 300)
        self.dropout = nn.Dropout(drop_p)
        self.num_layers = 2

        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=self.num_layers, dropout=drop_p, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=self.num_layers, dropout=drop_p, bidirectional=True)

        self.dense = nn.Linear(hidden_size*2*3, hidden_size)

        self.weight = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, seqence):
        batch_size = input_data_rec.shape[0]

        x_rec = self.word_embedding(input_data_rec)
        x_rep = self.word_embedding(input_data_rep)


        return pred