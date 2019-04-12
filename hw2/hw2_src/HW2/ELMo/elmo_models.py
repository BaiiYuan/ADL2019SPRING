import torch
import torch.nn as nn
import torch.nn.functional as F

from IPython import embed

device = "cuda" if torch.cuda.is_available() else "cpu"

class elmo_model(nn.Module):
    def __init__(self, input_size=300, classes=1, hidden_size=32,
                 drop_p=0.1, num_of_words=80000, out_of_words=80000):
        super(elmo_model, self).__init__()

        self.word_embedding = torch.nn.Embedding(num_of_words, 300)
        self.dropout = nn.Dropout(drop_p)
        self.num_layers = 1
        self.out_of_words = out_of_words

        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_size*2, hidden_size, batch_first=True, bidirectional=True)

        self.dense = nn.Linear(hidden_size*2, out_of_words)


    def forward(self, seqence):
        batch_size = seqence.shape[0]

        embedding = self.word_embedding(seqence)

        lstm1_output, _ = self.lstm1(embedding)
        lstm2_output, _ = self.lstm2(lstm1_output)

        last_hidden = lstm2_output[:, -1]
        pred = self.dense(last_hidden)

        # AdaptiveLogSoftmaxWithLoss

        embed()

        return pred