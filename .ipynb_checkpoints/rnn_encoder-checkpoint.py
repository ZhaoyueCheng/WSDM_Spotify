import torch.nn as nn
import torch
from models.layers import StackedBRNN
import torch.nn.functional as F


class RNNEncoder(nn.Module):
    def __init__(self):
        super(RNNEncoder, self).__init__()
        
        self.song_embedding = nn.Embedding(num_embeddings=3706389, embedding_dim=8, padding_idx=0)
        self.skip_embedding = nn.Embedding(num_embeddings=4, embedding_dim=2, padding_idx=0)
        self.bidirectional_rnn = StackedBRNN(input_size=10, hidden_size=int(20 / 2), num_layers=2, dropout_rate=0.0, rnn_type=nn.LSTM, concat_layers=False)
        self.linear1 = nn.Linear(20, 1)

    def forward(self, x_song, x_skip, x_mask):
        song_vec = self.song_embedding(x_song)
        skip_vec = self.skip_embedding(x_skip)
        rnn_input = torch.cat((song_vec, skip_vec), dim=2)
        rnn_output = self.bidirectional_rnn.forward(x=rnn_input, x_mask=x_mask)
        logits = F.sigmoid((self.linear1(rnn_output)).squeeze(-1))

        return logits

    def init_embeddings(self, embedding_matrix):
        self.song_embedding.weight.data.copy_(embedding_matrix)