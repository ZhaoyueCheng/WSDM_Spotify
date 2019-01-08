import torch.nn as nn
import torch
from models.layers import *
import torch.nn.functional as F


class RNNModel(nn.Module):
    def __init__(self, args, song_mat):
        super(RNNModel, self).__init__()

        # self.song_embedding = nn.Embedding(num_embeddings=args.num_songs, embedding_dim=args.song_embedding_dim, padding_idx=0)
        self.song_embedding = nn.Embedding.from_pretrained(song_mat, freeze=False, padding_idx=0)

        # self.optional_song_embedding = nn.Embedding(num_embeddings=args.num_songs, embedding_dim=28, padding_idx=0)
        self.history_rnn = StackedBRNN(input_size=args.song_embedding_dim + args.feature_dim,
                                       hidden_size=int(args.encoder_rnn_hidden_dim / 2),
                                       num_layers=args.encoder_rnn_layers, dropout_rate=args.encoder_rnn_dropout,
                                       rnn_type=nn.LSTM, concat_layers=False, dropout_output=False)
        self.predict_rnn = StackedBRNN(input_size=args.song_embedding_dim,
                                       hidden_size=int(args.encoder_rnn_hidden_dim / 2),
                                       num_layers=args.encoder_rnn_layers, dropout_rate=args.encoder_rnn_dropout,
                                       rnn_type=nn.LSTM, concat_layers=False, dropout_output=False)
        self.combine_rnn = StackedBRNN(input_size=args.encoder_rnn_hidden_dim,
                                       hidden_size=int(args.encoder_rnn_hidden_dim / 2),
                                       num_layers=args.encoder_rnn_layers, dropout_rate=args.encoder_rnn_dropout,
                                       rnn_type=nn.LSTM, concat_layers=False, dropout_output=False)
        self.ff = FeedForwardNetwork(input_size=args.encoder_rnn_hidden_dim, hidden_size=int(args.encoder_rnn_hidden_dim/2), output_size=1, dropout_rate=args.feed_forward_dropout)


    def forward(self, ht, hf, hm, pt, pm):
        song_vec_history = self.song_embedding(ht)
        song_vec_predict = self.song_embedding(pt,)
        # song_vec_opt = self.optional_song_embedding(x_song)
        history_input = torch.cat((song_vec_history, hf), dim=2)
        rnn_output_history = self.history_rnn.forward(x=history_input, x_mask=hm)
        rnn_output_predict = self.predict_rnn(x=song_vec_predict, x_mask = pm)

        rnn_output_concat = torch.cat((rnn_output_history, rnn_output_predict), dim=1)
        combine_mask = torch.cat((hm, pm), dim=1)

        rnn_output_combine = self.combine_rnn(x=rnn_output_concat, x_mask=combine_mask)

        predict_batch_len = pm.size(1)
        logits = F.sigmoid(self.ff.forward(rnn_output_combine).squeeze(-1))[:, -predict_batch_len:]

        return logits

    # def init_embeddings(self, embedding_matrix):
    #     self.song_embedding.weight.data.copy_(embedding_matrix)


class RNNModelSelfAttn(nn.Module):
    def __init__(self, args):
        super(RNNModelSelfAttn, self).__init__()

        self.song_embedding = nn.Embedding(num_embeddings=args.num_songs, embedding_dim=args.song_embedding_dim, padding_idx=0)
        # self.optional_song_embedding = nn.Embedding(num_embeddings=args.num_songs, embedding_dim=28, padding_idx=0)
        self.history_rnn = StackedBRNN(input_size=args.song_embedding_dim + args.feature_dim,
                                       hidden_size=int(args.encoder_rnn_hidden_dim / 2),
                                       num_layers=args.encoder_rnn_layers, dropout_rate=args.encoder_rnn_dropout,
                                       rnn_type=nn.LSTM, concat_layers=False, dropout_output=False)
        self.predict_rnn = StackedBRNN(input_size=args.song_embedding_dim,
                                       hidden_size=int(args.encoder_rnn_hidden_dim / 2),
                                       num_layers=args.encoder_rnn_layers, dropout_rate=args.encoder_rnn_dropout,
                                       rnn_type=nn.LSTM, concat_layers=False, dropout_output=False)
        self.combine_rnn = StackedBRNN(input_size=args.encoder_rnn_hidden_dim,
                                       hidden_size=int(args.encoder_rnn_hidden_dim / 2),
                                       num_layers=args.encoder_rnn_layers, dropout_rate=args.encoder_rnn_dropout,
                                       rnn_type=nn.LSTM, concat_layers=False, dropout_output=False)
        self.self_attn = SelfAttnMatch(args.encoder_rnn_hidden_dim, identity=False)

        self.ff1 = FeedForwardNetwork(input_size=args.encoder_rnn_hidden_dim * 2,
                                      hidden_size=int(args.encoder_rnn_hidden_dim),
                                      output_size=args.encoder_rnn_hidden_dim, dropout_rate=args.feed_forward_dropout)
        self.ff2 = FeedForwardNetwork(input_size=args.encoder_rnn_hidden_dim,
                                      hidden_size=int(args.encoder_rnn_hidden_dim / 2),
                                      output_size=1, dropout_rate=args.feed_forward_dropout)


    def forward(self, ht, hf, hm, pt, pm):
        song_vec_history = self.song_embedding(ht)
        song_vec_predict = self.song_embedding(pt)
        # song_vec_opt = self.optional_song_embedding(x_song)
        history_input = torch.cat((song_vec_history, hf), dim=2)
        rnn_output_history = self.history_rnn.forward(x=history_input, x_mask=hm)
        rnn_output_predict = self.predict_rnn(x=song_vec_predict, x_mask = pm)

        rnn_output_concat = torch.cat((rnn_output_history, rnn_output_predict), dim=1)
        combine_mask = torch.cat((hm, pm), dim = 1)

        rnn_output_combine = self.combine_rnn(x=rnn_output_concat, x_mask=combine_mask)
        self_attn_combine = self.self_attn(x=rnn_output_combine, x_mask=combine_mask)

        predict_batch_len = pm.size(1)

        ff1_result = F.relu(self.ff1(torch.cat([rnn_output_combine, self_attn_combine], dim=2)))

        logits = F.sigmoid(self.ff2(ff1_result).squeeze(-1))[:, -predict_batch_len:]

        return logits

    def init_embeddings(self, embedding_matrix):
        self.song_embedding.weight.data.copy_(embedding_matrix)


class RNNModelAtt(nn.Module):
    def __init__(self, args):
        super(RNNModelAtt, self).__init__()

        self.song_embedding = nn.Embedding(num_embeddings=args.num_songs, embedding_dim=args.song_embedding_dim, padding_idx=0)
        # self.optional_song_embedding = nn.Embedding(num_embeddings=args.num_songs, embedding_dim=28, padding_idx=0)
        self.history_rnn = StackedBRNN(input_size=args.song_embedding_dim + args.feature_dim,
                                       hidden_size=int(args.encoder_rnn_hidden_dim / 2),
                                       num_layers=args.encoder_rnn_layers, dropout_rate=args.encoder_rnn_dropout,
                                       rnn_type=nn.LSTM, concat_layers=False, dropout_output=False)
        self.predict_rnn = StackedBRNN(input_size=args.song_embedding_dim,
                                       hidden_size=int(args.encoder_rnn_hidden_dim / 2),
                                       num_layers=args.encoder_rnn_layers, dropout_rate=args.encoder_rnn_dropout,
                                       rnn_type=nn.LSTM, concat_layers=False, dropout_output=False)
        # self.combine_rnn = StackedBRNN(input_size=args.encoder_rnn_hidden_dim,
        #                                hidden_size=int(args.encoder_rnn_hidden_dim / 2),
        #                                num_layers=args.encoder_rnn_layers, dropout_rate=args.encoder_rnn_dropout,
        #                                rnn_type=nn.LSTM, concat_layers=False, dropout_output=False)
        self.history_attn = SeqAttnMatch(args.encoder_rnn_hidden_dim, identity=False)
        self.history_attn_gate = Gate(args.encoder_rnn_hidden_dim*2)
        self.history_attn_rnn = StackedBRNN(input_size=args.encoder_rnn_hidden_dim*2, hidden_size=int(args.encoder_rnn_hidden_dim / 2),
                                            num_layers=args.encoder_rnn_layers, dropout_rate=args.encoder_rnn_dropout,
                                            rnn_type=nn.LSTM, concat_layers=False, dropout_output=False)

        self.predict_self_attn = SelfAttnMatch(args.encoder_rnn_hidden_dim, identity=False)
        self.predict_attn_gate = Gate(args.encoder_rnn_hidden_dim * 2)
        self.predict_attn_rnn = StackedBRNN(input_size=args.encoder_rnn_hidden_dim*2, hidden_size=int(args.encoder_rnn_hidden_dim / 2),
                                            num_layers=args.encoder_rnn_layers, dropout_rate=args.encoder_rnn_dropout,
                                            rnn_type=nn.LSTM, concat_layers=False, dropout_output=False)

        self.ff = FeedForwardNetwork(input_size=args.encoder_rnn_hidden_dim, hidden_size=int(args.encoder_rnn_hidden_dim/2), output_size=1, dropout_rate=args.feed_forward_dropout)


    def forward(self, ht, hf, hm, pt, pm):
        song_vec_history = self.song_embedding(ht)
        song_vec_predict = self.song_embedding(pt,)
        # song_vec_opt = self.optional_song_embedding(x_song)
        history_input = torch.cat((song_vec_history, hf), dim=2)
        rnn_output_history = self.history_rnn.forward(x=history_input, x_mask=hm)
        rnn_output_predict = self.predict_rnn(x=song_vec_predict, x_mask = pm)

        # rnn_output_concat = torch.cat((rnn_output_history, rnn_output_predict), dim=1)
        # combine_mask = torch.cat((hm, pm), dim = 1)
        # rnn_output_combine = self.combine_rnn(x=rnn_output_concat, x_mask = combine_mask)

        # match history to predict
        history_attn_predict = self.history_attn(rnn_output_predict, rnn_output_history, hm)
        history_attn_rnn_input = self.history_attn_gate(torch.cat([rnn_output_predict, history_attn_predict], dim=2))
        hist_attn_predict_encoded = self.history_attn_rnn(history_attn_rnn_input, pm)

        # match predict to themselves
        predict_self_attn_hiddens = self.predict_self_attn(hist_attn_predict_encoded, pm)
        predict_attn_rnn_input = self.predict_attn_gate(torch.cat([hist_attn_predict_encoded, predict_self_attn_hiddens], dim=2))
        predict_attn_rnn_output = self.predict_attn_rnn(predict_attn_rnn_input, pm)

        predict_batch_len = pm.size(1)
        logits = F.sigmoid(self.ff.forward(predict_attn_rnn_output).squeeze(-1))[:, -predict_batch_len:]

        return logits

    def init_embeddings(self, embedding_matrix):
        self.song_embedding.weight.data.copy_(embedding_matrix)


class RNNModelAtt1(nn.Module):
    def __init__(self, args):
        super(RNNModelAtt1, self).__init__()

        self.song_embedding = nn.Embedding(num_embeddings=args.num_songs, embedding_dim=args.song_embedding_dim, padding_idx=0)
        # self.optional_song_embedding = nn.Embedding(num_embeddings=args.num_songs, embedding_dim=28, padding_idx=0)
        self.history_rnn = StackedBRNN(input_size=args.song_embedding_dim + args.feature_dim,
                                       hidden_size=int(args.encoder_rnn_hidden_dim / 2),
                                       num_layers=args.encoder_rnn_layers, dropout_rate=args.encoder_rnn_dropout,
                                       rnn_type=nn.LSTM, concat_layers=False, dropout_output=False)
        self.predict_rnn = StackedBRNN(input_size=args.song_embedding_dim,
                                       hidden_size=int(args.encoder_rnn_hidden_dim / 2),
                                       num_layers=args.encoder_rnn_layers, dropout_rate=args.encoder_rnn_dropout,
                                       rnn_type=nn.LSTM, concat_layers=False, dropout_output=False)

        self.history_attn_predict = SeqAttnMatch(args.encoder_rnn_hidden_dim, identity=False)
        self.history_attn_self = SelfAttnMatch(args.encoder_rnn_hidden_dim, identity=False)

        self.predict_attn_history = SeqAttnMatch(args.encoder_rnn_hidden_dim, identity=False)
        self.predict_attn_self = SelfAttnMatch(args.encoder_rnn_hidden_dim, identity=False)

        self.combine_rnn = StackedBRNN(input_size=args.encoder_rnn_hidden_dim*3,
                                       hidden_size=int(args.encoder_rnn_hidden_dim),
                                       num_layers=args.encoder_rnn_layers, dropout_rate=args.encoder_rnn_dropout,
                                       rnn_type=nn.LSTM, concat_layers=False, dropout_output=False)

        self.ff1 = FeedForwardNetwork(input_size=args.encoder_rnn_hidden_dim * 2,
                                      hidden_size=int(args.encoder_rnn_hidden_dim),
                                      output_size=int(args.encoder_rnn_hidden_dim), dropout_rate=args.feed_forward_dropout)
        self.ff2 = FeedForwardNetwork(input_size=args.encoder_rnn_hidden_dim,
                                      hidden_size=int(args.encoder_rnn_hidden_dim / 2),
                                      output_size=1, dropout_rate=args.feed_forward_dropout)


    def forward(self, ht, hf, hm, pt, pm):
        song_vec_history = self.song_embedding(ht)
        song_vec_predict = self.song_embedding(pt)
        # song_vec_opt = self.optional_song_embedding(x_song)
        history_input = torch.cat((song_vec_history, hf), dim=2)
        rnn_output_history = self.history_rnn.forward(x=history_input, x_mask=hm)
        rnn_output_predict = self.predict_rnn(x=song_vec_predict, x_mask = pm)

        # rnn_output_concat = torch.cat((rnn_output_history, rnn_output_predict), dim=1)
        # combine_mask = torch.cat((hm, pm), dim = 1)
        # rnn_output_combine = self.combine_rnn(x=rnn_output_concat, x_mask = combine_mask)

        # match history to predict
        history_attn_predict = self.history_attn_predict(rnn_output_history, rnn_output_predict, pm)
        history_attn_self = self.history_attn_self(rnn_output_history, hm)

        # match predict to themselves
        predict_attn_history = self.predict_attn_history(rnn_output_predict, rnn_output_history, hm)
        predict_attn_self = self.predict_attn_self(rnn_output_predict, pm)

        history_input = torch.cat((rnn_output_history, history_attn_predict, history_attn_self), dim=2)
        predict_input = torch.cat((rnn_output_predict, predict_attn_history, predict_attn_self), dim=2)

        combine_rnn_input = torch.cat((history_input, predict_input), dim=1)
        combine_mask = torch.cat((hm, pm), dim = 1)
        rnn_output_combine = self.combine_rnn(x = combine_rnn_input, x_mask = combine_mask)

        predict_batch_len = pm.size(1)

        ff1_result = F.relu(self.ff1(rnn_output_combine))

        logits = F.sigmoid(self.ff2(ff1_result).squeeze(-1))[:, -predict_batch_len:]

        return logits

    def init_embeddings(self, embedding_matrix):
        self.song_embedding.weight.data.copy_(embedding_matrix)

class RNNModelAtt1_Double_LSTM(nn.Module):
    def __init__(self, args):
        super(RNNModelAtt1_Double_LSTM, self).__init__()

        self.song_embedding = nn.Embedding(num_embeddings=args.num_songs, embedding_dim=args.song_embedding_dim, padding_idx=0)
        # self.optional_song_embedding = nn.Embedding(num_embeddings=args.num_songs, embedding_dim=28, padding_idx=0)
        self.history_rnn = StackedBRNN(input_size=args.song_embedding_dim + args.feature_dim,
                                       hidden_size=int(args.encoder_rnn_hidden_dim / 2),
                                       num_layers=2, dropout_rate=args.encoder_rnn_dropout,
                                       rnn_type=nn.LSTM, concat_layers=False, dropout_output=False)
        self.predict_rnn = StackedBRNN(input_size=args.song_embedding_dim,
                                       hidden_size=int(args.encoder_rnn_hidden_dim / 2),
                                       num_layers=2, dropout_rate=args.encoder_rnn_dropout,
                                       rnn_type=nn.LSTM, concat_layers=False, dropout_output=False)

        self.history_attn_predict = SeqAttnMatch(args.encoder_rnn_hidden_dim, identity=False)
        self.history_attn_self = SelfAttnMatch(args.encoder_rnn_hidden_dim, identity=False)

        self.predict_attn_history = SeqAttnMatch(args.encoder_rnn_hidden_dim, identity=False)
        self.predict_attn_self = SelfAttnMatch(args.encoder_rnn_hidden_dim, identity=False)

        self.combine_rnn = StackedBRNN(input_size=args.encoder_rnn_hidden_dim*3,
                                       hidden_size=int(args.encoder_rnn_hidden_dim),
                                       num_layers=2, dropout_rate=args.encoder_rnn_dropout,
                                       rnn_type=nn.LSTM, concat_layers=False, dropout_output=False)

        self.ff1 = FeedForwardNetwork(input_size=args.encoder_rnn_hidden_dim * 2,
                                      hidden_size=int(args.encoder_rnn_hidden_dim),
                                      output_size=int(args.encoder_rnn_hidden_dim), dropout_rate=args.feed_forward_dropout)
        self.ff2 = FeedForwardNetwork(input_size=args.encoder_rnn_hidden_dim,
                                      hidden_size=int(args.encoder_rnn_hidden_dim / 2),
                                      output_size=1, dropout_rate=args.feed_forward_dropout)

        self.dropout_rate = args.feed_forward_dropout

    class RNNModelAtt1SeperateEncoder(nn.Module):
        def __init__(self, args):
            super(RNNModelAtt1SeperateEncoder, self).__init__()

            self.song_embedding = nn.Embedding(num_embeddings=args.num_songs, embedding_dim=args.song_embedding_dim,
                                               padding_idx=0)
            self.history_rnn = StackedBRNN(input_size=args.song_embedding_dim + args.feature_dim,
                                           hidden_size=int(args.encoder_rnn_hidden_dim / 2),
                                           num_layers=args.encoder_rnn_layers, dropout_rate=args.encoder_rnn_dropout,
                                           rnn_type=nn.LSTM, concat_layers=False, dropout_output=False)
            self.predict_rnn = StackedBRNN(input_size=args.song_embedding_dim,
                                           hidden_size=int(args.encoder_rnn_hidden_dim / 2),
                                           num_layers=args.encoder_rnn_layers, dropout_rate=args.encoder_rnn_dropout,
                                           rnn_type=nn.LSTM, concat_layers=False, dropout_output=False)

            self.history_attn_predict = SeqAttnMatch(args.encoder_rnn_hidden_dim, identity=False)
            self.history_attn_self = SelfAttnMatch(args.encoder_rnn_hidden_dim, identity=False)

            self.predict_attn_history = SeqAttnMatch(args.encoder_rnn_hidden_dim, identity=False)
            self.predict_attn_self = SelfAttnMatch(args.encoder_rnn_hidden_dim, identity=False)

            self.combine_rnn = StackedBRNN(input_size=args.encoder_rnn_hidden_dim * 3,
                                           hidden_size=int(args.encoder_rnn_hidden_dim),
                                           num_layers=args.encoder_rnn_layers, dropout_rate=args.encoder_rnn_dropout,
                                           rnn_type=nn.LSTM, concat_layers=False, dropout_output=False)

            self.combine_attn_self = SelfAttnMatch(args.encoder_rnn_hidden_dim * 2, identity=False)

            self.ff1 = FeedForwardNetwork(input_size=int(args.encoder_rnn_hidden_dim) * 4,
                                          hidden_size=int(args.encoder_rnn_hidden_dim) * 2,
                                          output_size=int(args.encoder_rnn_hidden_dim),
                                          dropout_rate=args.feed_forward_dropout)

            self.ff2 = FeedForwardNetwork(input_size=args.encoder_rnn_hidden_dim,
                                          hidden_size=int(args.encoder_rnn_hidden_dim / 2),
                                          output_size=1, dropout_rate=args.feed_forward_dropout)

            self.dropout_rate = args.feed_forward_dropout

        def forward(self, ht, hf, hm, pt, pm):
            song_vec_history = self.song_embedding(ht)
            song_vec_predict = self.song_embedding(pt, )
            history_input = torch.cat((song_vec_history, hf), dim=2)
            rnn_output_history = self.history_rnn(x=history_input, x_mask=hm)
            rnn_output_predict = self.predict_rnn(x=song_vec_predict, x_mask=pm)

            # match history to predict
            history_attn_predict = self.history_attn_predict(rnn_output_history, rnn_output_predict, pm)
            history_attn_self = self.history_attn_self(rnn_output_history, hm)

            # match predict to themselves
            predict_attn_history = self.predict_attn_history(rnn_output_predict, rnn_output_history, hm)
            predict_attn_self = self.predict_attn_self(rnn_output_predict, pm)

            history_input = torch.cat((rnn_output_history, history_attn_predict, history_attn_self), dim=2)
            predict_input = torch.cat((rnn_output_predict, predict_attn_history, predict_attn_self), dim=2)

            combine_rnn_input = torch.cat((history_input, predict_input), dim=1)
            combine_mask = torch.cat((hm, pm), dim=1)
            rnn_output_combine = self.combine_rnn(x=combine_rnn_input, x_mask=combine_mask)

            # after combine, self attention over the combined model
            attn_output_combine = self.combine_attn_self(rnn_output_combine, combine_mask)

            # final representation
            final_repr = torch.cat((rnn_output_combine, attn_output_combine), dim=2)

            predict_batch_len = pm.size(1)

            ff1_result = F.dropout(F.relu(self.ff1(final_repr)), p=self.dropout_rate, training=self.training)

            logits = F.sigmoid(self.ff2(ff1_result).squeeze(-1))[:, -predict_batch_len:]

            return logits

        def init_embeddings(self, embedding_matrix):
            self.song_embedding.weight.data.copy_(embedding_matrix)

    def forward(self, ht, hf, hm, pt, pm):
        song_vec_history = self.song_embedding(ht)
        song_vec_predict = self.song_embedding(pt)
        # song_vec_opt = self.optional_song_embedding(x_song)
        history_input = torch.cat((song_vec_history, hf), dim=2)
        rnn_output_history = self.history_rnn.forward(x=history_input, x_mask=hm)
        rnn_output_predict = self.predict_rnn(x=song_vec_predict, x_mask = pm)

        # rnn_output_concat = torch.cat((rnn_output_history, rnn_output_predict), dim=1)
        # combine_mask = torch.cat((hm, pm), dim = 1)
        # rnn_output_combine = self.combine_rnn(x=rnn_output_concat, x_mask = combine_mask)

        # match history to predict
        history_attn_predict = self.history_attn_predict(rnn_output_history, rnn_output_predict, pm)
        history_attn_self = self.history_attn_self(rnn_output_history, hm)

        # match predict to themselves
        predict_attn_history = self.predict_attn_history(rnn_output_predict, rnn_output_history, hm)
        predict_attn_self = self.predict_attn_self(rnn_output_predict, pm)

        history_input = torch.cat((rnn_output_history, history_attn_predict, history_attn_self), dim=2)
        predict_input = torch.cat((rnn_output_predict, predict_attn_history, predict_attn_self), dim=2)

        combine_rnn_input = torch.cat((history_input, predict_input), dim=1)
        combine_mask = torch.cat((hm, pm), dim = 1)
        rnn_output_combine = self.combine_rnn(x = combine_rnn_input, x_mask = combine_mask)

        predict_batch_len = pm.size(1)

        ff1_result = F.dropout(F.relu(self.ff1(rnn_output_combine)), p=self.dropout_rate, training=self.training)

        logits = F.sigmoid(self.ff2(ff1_result).squeeze(-1))[:, -predict_batch_len:]

        return logits

    def init_embeddings(self, embedding_matrix):
        self.song_embedding.weight.data.copy_(embedding_matrix)


class RNNModelAtt1SeperateEncoder_Double_LSTM(nn.Module):
    def __init__(self, args):
        super(RNNModelAtt1SeperateEncoder_Double_LSTM, self).__init__()

        self.song_embedding = nn.Embedding(num_embeddings=args.num_songs, embedding_dim=args.song_embedding_dim, padding_idx=0)
        self.history_rnn = StackedBRNN(input_size=args.song_embedding_dim + args.feature_dim,
                                       hidden_size=int(args.encoder_rnn_hidden_dim / 2),
                                       num_layers=2, dropout_rate=args.encoder_rnn_dropout,
                                       rnn_type=nn.LSTM, concat_layers=False, dropout_output=False)
        self.predict_rnn = StackedBRNN(input_size=args.song_embedding_dim,
                                       hidden_size=int(args.encoder_rnn_hidden_dim / 2),
                                       num_layers=2, dropout_rate=args.encoder_rnn_dropout,
                                       rnn_type=nn.LSTM, concat_layers=False, dropout_output=False)

        self.history_attn_predict = SeqAttnMatch(args.encoder_rnn_hidden_dim, identity=False)
        self.history_attn_self = SelfAttnMatch(args.encoder_rnn_hidden_dim, identity=False)

        self.predict_attn_history = SeqAttnMatch(args.encoder_rnn_hidden_dim, identity=False)
        self.predict_attn_self = SelfAttnMatch(args.encoder_rnn_hidden_dim, identity=False)

        self.combine_rnn = StackedBRNN(input_size=args.encoder_rnn_hidden_dim*3,
                                       hidden_size=int(args.encoder_rnn_hidden_dim),
                                       num_layers=2, dropout_rate=args.encoder_rnn_dropout,
                                       rnn_type=nn.LSTM, concat_layers=False, dropout_output=False)

        self.combine_attn_self = SelfAttnMatch(args.encoder_rnn_hidden_dim*2, identity=False)

        self.ff1 = FeedForwardNetwork(input_size=int(args.encoder_rnn_hidden_dim) * 4,
                                      hidden_size=int(args.encoder_rnn_hidden_dim) * 2,
                                      output_size=int(args.encoder_rnn_hidden_dim), dropout_rate=args.feed_forward_dropout)

        self.ff2 = FeedForwardNetwork(input_size=args.encoder_rnn_hidden_dim,
                                      hidden_size=int(args.encoder_rnn_hidden_dim / 2),
                                      output_size=1, dropout_rate=args.feed_forward_dropout)

        self.dropout_rate = args.feed_forward_dropout


    def forward(self, ht, hf, hm, pt, pm):

        song_vec_history = self.song_embedding(ht)
        song_vec_predict = self.song_embedding(pt)
        history_input = torch.cat((song_vec_history, hf), dim=2)
        rnn_output_history = self.history_rnn(x=history_input, x_mask=hm)
        rnn_output_predict = self.predict_rnn(x=song_vec_predict, x_mask = pm)

        # match history to predict
        history_attn_predict = self.history_attn_predict(rnn_output_history, rnn_output_predict, pm)
        history_attn_self = self.history_attn_self(rnn_output_history, hm)

        # match predict to themselves
        predict_attn_history = self.predict_attn_history(rnn_output_predict, rnn_output_history, hm)
        predict_attn_self = self.predict_attn_self(rnn_output_predict, pm)

        history_input = torch.cat((rnn_output_history, history_attn_predict, history_attn_self), dim=2)
        predict_input = torch.cat((rnn_output_predict, predict_attn_history, predict_attn_self), dim=2)

        combine_rnn_input = torch.cat((history_input, predict_input), dim=1)
        combine_mask = torch.cat((hm, pm), dim = 1)
        rnn_output_combine = self.combine_rnn(x = combine_rnn_input, x_mask = combine_mask)

        # after combine, self attention over the combined model
        attn_output_combine = self.combine_attn_self(rnn_output_combine, combine_mask)

        # final representation
        final_repr = torch.cat((rnn_output_combine, attn_output_combine), dim=2)

        predict_batch_len = pm.size(1)

        ff1_result = F.dropout(F.relu(self.ff1(final_repr)), p=self.dropout_rate, training=self.training)

        logits = F.sigmoid(self.ff2(ff1_result).squeeze(-1))[:, -predict_batch_len:]

        return logits

    def init_embeddings(self, embedding_matrix):
        self.song_embedding.weight.data.copy_(embedding_matrix)


class RNNModelAtt1ShareEncoder(nn.Module):
    def __init__(self, args):
        super(RNNModelAtt1ShareEncoder, self).__init__()

        self.song_embedding = nn.Embedding(num_embeddings=args.num_songs, embedding_dim=args.song_embedding_dim, padding_idx=0)

        self.song_rnn = StackedBRNN(input_size=args.song_embedding_dim,
                                    hidden_size=int(args.encoder_rnn_hidden_dim / 2),
                                    num_layers=1, dropout_rate=args.encoder_rnn_dropout,
                                    rnn_type=nn.LSTM, concat_layers=False, dropout_output=False)

        self.history_feature_rnn = StackedBRNN(input_size=args.feature_dim,
                                       hidden_size=int(args.encoder_rnn_hidden_dim / 2),
                                       num_layers=1, dropout_rate=args.encoder_rnn_dropout,
                                       rnn_type=nn.LSTM, concat_layers=False, dropout_output=False)

        self.history_pwnn = PositionwiseNN(idim=args.encoder_rnn_hidden_dim * 2,
                                                 hdim=args.encoder_rnn_hidden_dim,
                                                 dropout_rate=args.feed_forward_dropout)

        self.history_attn_predict = SeqAttnMatch(args.encoder_rnn_hidden_dim, identity=False)
        self.history_attn_self = SelfAttnMatch(args.encoder_rnn_hidden_dim, identity=False)

        self.predict_attn_history = SeqAttnMatch(args.encoder_rnn_hidden_dim, identity=False)
        self.predict_attn_self = SelfAttnMatch(args.encoder_rnn_hidden_dim, identity=False)

        self.combine_rnn = StackedBRNN(input_size=args.encoder_rnn_hidden_dim*3,
                                       hidden_size=int(args.encoder_rnn_hidden_dim),
                                       num_layers=args.encoder_rnn_layers, dropout_rate=args.encoder_rnn_dropout,
                                       rnn_type=nn.LSTM, concat_layers=False, dropout_output=False)

        self.combine_attn_self = SelfAttnMatch(args.encoder_rnn_hidden_dim*2, identity=False)

        self.ff1 = FeedForwardNetwork(input_size=int(args.encoder_rnn_hidden_dim) * 4,
                                      hidden_size=int(args.encoder_rnn_hidden_dim) * 2,
                                      output_size=int(args.encoder_rnn_hidden_dim), dropout_rate=args.feed_forward_dropout)

        self.ff2 = FeedForwardNetwork(input_size=args.encoder_rnn_hidden_dim,
                                      hidden_size=int(args.encoder_rnn_hidden_dim / 2),
                                      output_size=1, dropout_rate=args.feed_forward_dropout)

        self.dropout_rate = args.feed_forward_dropout


    def forward(self, ht, hf, hm, pt, pm):
        song_vec_history = self.song_embedding(ht)
        song_vec_predict = self.song_embedding(pt)

        predict_encoded = self.song_rnn(x=song_vec_predict, x_mask=pm).contiguous()

        song_rnn_history = self.song_rnn(x=song_vec_history, x_mask=hm)
        feature_rnn_history = self.history_feature_rnn(x=hf, x_mask=hm)
        history_pwnn_input = torch.cat((song_rnn_history, feature_rnn_history), dim=2)
        history_encoded = self.history_pwnn(history_pwnn_input).contiguous()


        # attend history
        history_attn_predict = self.history_attn_predict(history_encoded, predict_encoded, pm)
        history_attn_self = self.history_attn_self(history_encoded, hm)

        # attend predict
        predict_attn_history = self.predict_attn_history(predict_encoded, history_encoded, hm)
        predict_attn_self = self.predict_attn_self(predict_encoded, pm)

        # combine using RNN
        history_input = torch.cat((history_encoded, history_attn_predict, history_attn_self), dim=2)
        predict_input = torch.cat((predict_encoded, predict_attn_history, predict_attn_self), dim=2)

        combine_rnn_input = torch.cat((history_input, predict_input), dim=1)
        combine_mask = torch.cat((hm, pm), dim=1)
        rnn_output_combine = self.combine_rnn(x=combine_rnn_input, x_mask=combine_mask)

        # after combine, self attention over the combined model
        attn_output_combine = self.combine_attn_self(rnn_output_combine, combine_mask)

        # final representation
        final_repr = torch.cat((rnn_output_combine, attn_output_combine), dim=2)

        predict_batch_len = pm.size(1)

        ff1_result = F.dropout(F.relu(self.ff1(final_repr)), p=self.dropout_rate, training=self.training)

        logits = F.sigmoid(self.ff2(ff1_result).squeeze(-1))[:, -predict_batch_len:]

        return logits

    def init_embeddings(self, embedding_matrix):
        self.song_embedding.weight.data.copy_(embedding_matrix)


class RNNModelAtt1SeperateEncoder(nn.Module):
    def __init__(self, args):
        super(RNNModelAtt1SeperateEncoder, self).__init__()

        self.song_embedding = nn.Embedding(num_embeddings=args.num_songs, embedding_dim=args.song_embedding_dim, padding_idx=0)
        self.history_rnn = StackedBRNN(input_size=args.song_embedding_dim + args.feature_dim,
                                       hidden_size=int(args.encoder_rnn_hidden_dim / 2),
                                       num_layers=args.encoder_rnn_layers, dropout_rate=args.encoder_rnn_dropout,
                                       rnn_type=nn.LSTM, concat_layers=False, dropout_output=False)
        self.predict_rnn = StackedBRNN(input_size=args.song_embedding_dim,
                                       hidden_size=int(args.encoder_rnn_hidden_dim / 2),
                                       num_layers=args.encoder_rnn_layers, dropout_rate=args.encoder_rnn_dropout,
                                       rnn_type=nn.LSTM, concat_layers=False, dropout_output=False)

        self.history_attn_predict = SeqAttnMatch(args.encoder_rnn_hidden_dim, identity=False)
        self.history_attn_self = SelfAttnMatch(args.encoder_rnn_hidden_dim, identity=False)

        self.predict_attn_history = SeqAttnMatch(args.encoder_rnn_hidden_dim, identity=False)
        self.predict_attn_self = SelfAttnMatch(args.encoder_rnn_hidden_dim, identity=False)

        self.combine_rnn = StackedBRNN(input_size=args.encoder_rnn_hidden_dim*3,
                                       hidden_size=int(args.encoder_rnn_hidden_dim),
                                       num_layers=args.encoder_rnn_layers, dropout_rate=args.encoder_rnn_dropout,
                                       rnn_type=nn.LSTM, concat_layers=False, dropout_output=False)

        self.combine_attn_self = SelfAttnMatch(args.encoder_rnn_hidden_dim*2, identity=False)

        self.ff1 = FeedForwardNetwork(input_size=int(args.encoder_rnn_hidden_dim) * 4,
                                      hidden_size=int(args.encoder_rnn_hidden_dim) * 2,
                                      output_size=int(args.encoder_rnn_hidden_dim), dropout_rate=args.feed_forward_dropout)

        self.ff2 = FeedForwardNetwork(input_size=args.encoder_rnn_hidden_dim,
                                      hidden_size=int(args.encoder_rnn_hidden_dim / 2),
                                      output_size=1, dropout_rate=args.feed_forward_dropout)

        self.dropout_rate = args.feed_forward_dropout


    def forward(self, ht, hf, hm, pt, pm):

        song_vec_history = self.song_embedding(ht)
        song_vec_predict = self.song_embedding(pt,)
        history_input = torch.cat((song_vec_history, hf), dim=2)
        rnn_output_history = self.history_rnn(x=history_input, x_mask=hm)
        rnn_output_predict = self.predict_rnn(x=song_vec_predict, x_mask = pm)

        # match history to predict
        history_attn_predict = self.history_attn_predict(rnn_output_history, rnn_output_predict, pm)
        history_attn_self = self.history_attn_self(rnn_output_history, hm)

        # match predict to themselves
        predict_attn_history = self.predict_attn_history(rnn_output_predict, rnn_output_history, hm)
        predict_attn_self = self.predict_attn_self(rnn_output_predict, pm)

        history_input = torch.cat((rnn_output_history, history_attn_predict, history_attn_self), dim=2)
        predict_input = torch.cat((rnn_output_predict, predict_attn_history, predict_attn_self), dim=2)

        combine_rnn_input = torch.cat((history_input, predict_input), dim=1)
        combine_mask = torch.cat((hm, pm), dim = 1)
        rnn_output_combine = self.combine_rnn(x = combine_rnn_input, x_mask = combine_mask)

        # after combine, self attention over the combined model
        attn_output_combine = self.combine_attn_self(rnn_output_combine, combine_mask)

        # final representation
        final_repr = torch.cat((rnn_output_combine, attn_output_combine), dim=2)

        predict_batch_len = pm.size(1)

        ff1_result = F.dropout(F.relu(self.ff1(final_repr)), p=self.dropout_rate, training=self.training)

        logits = F.sigmoid(self.ff2(ff1_result).squeeze(-1))[:, -predict_batch_len:]

        return logits

    def init_embeddings(self, embedding_matrix):
        self.song_embedding.weight.data.copy_(embedding_matrix)


class RNNModelAtt2(nn.Module):
    def __init__(self, args):
        super(RNNModelAtt2, self).__init__()

        self.song_embedding = nn.Embedding(num_embeddings=args.num_songs, embedding_dim=args.song_embedding_dim, padding_idx=0)
        self.optional_song_embedding = nn.Embedding(num_embeddings=args.num_songs, embedding_dim=50, padding_idx=0)
        self.history_rnn = StackedBRNN(input_size=args.song_embedding_dim + args.feature_dim + 50,
                                       hidden_size=int(args.encoder_rnn_hidden_dim / 2),
                                       num_layers=args.encoder_rnn_layers, dropout_rate=args.encoder_rnn_dropout,
                                       rnn_type=nn.LSTM, concat_layers=False, dropout_output=False)
        self.predict_rnn = StackedBRNN(input_size=args.song_embedding_dim + 50,
                                       hidden_size=int(args.encoder_rnn_hidden_dim / 2),
                                       num_layers=args.encoder_rnn_layers, dropout_rate=args.encoder_rnn_dropout,
                                       rnn_type=nn.LSTM, concat_layers=False, dropout_output=False)

        self.history_attn_predict = SeqAttnMatch(args.encoder_rnn_hidden_dim, identity=False)
        self.history_attn_self = SelfAttnMatch(args.encoder_rnn_hidden_dim, identity=False)

        self.predict_attn_history = SeqAttnMatch(args.encoder_rnn_hidden_dim, identity=False)
        self.predict_attn_self = SelfAttnMatch(args.encoder_rnn_hidden_dim, identity=False)

        self.combine_rnn = StackedBRNN(input_size=args.encoder_rnn_hidden_dim*3,
                                       hidden_size=int(args.encoder_rnn_hidden_dim),
                                       num_layers=args.encoder_rnn_layers, dropout_rate=args.encoder_rnn_dropout,
                                       rnn_type=nn.LSTM, concat_layers=False, dropout_output=False)

        self.ff1 = FeedForwardNetwork(input_size=args.encoder_rnn_hidden_dim * 2,
                                      hidden_size=int(args.encoder_rnn_hidden_dim),
                                      output_size=int(args.encoder_rnn_hidden_dim), dropout_rate=args.feed_forward_dropout)
        self.ff2 = FeedForwardNetwork(input_size=args.encoder_rnn_hidden_dim,
                                      hidden_size=int(args.encoder_rnn_hidden_dim / 2),
                                      output_size=1, dropout_rate=args.feed_forward_dropout)


    def forward(self, ht, hf, hm, pt, pm):
        song_vec_org_history = self.song_embedding(ht)
        song_vec_org_predict = self.song_embedding(pt)

        song_vec_opt_history = self.optional_song_embedding(ht)
        song_vec_opt_predict = self.optional_song_embedding(pt)

        song_vec_history = torch.cat((song_vec_org_history, song_vec_opt_history), dim=2)
        song_vec_predict = torch.cat((song_vec_org_predict, song_vec_opt_predict), dim=2)

        history_input = torch.cat((song_vec_history, hf), dim=2)
        rnn_output_history = self.history_rnn.forward(x=history_input, x_mask=hm)
        rnn_output_predict = self.predict_rnn(x=song_vec_predict, x_mask = pm)

        # rnn_output_concat = torch.cat((rnn_output_history, rnn_output_predict), dim=1)
        # combine_mask = torch.cat((hm, pm), dim = 1)
        # rnn_output_combine = self.combine_rnn(x=rnn_output_concat, x_mask = combine_mask)

        # match history to predict
        history_attn_predict = self.history_attn_predict(rnn_output_history, rnn_output_predict, pm)
        history_attn_self = self.history_attn_self(rnn_output_history, hm)

        # match predict to themselves
        predict_attn_history = self.predict_attn_history(rnn_output_predict, rnn_output_history, hm)
        predict_attn_self = self.predict_attn_self(rnn_output_predict, pm)

        history_input = torch.cat((rnn_output_history, history_attn_predict, history_attn_self), dim=2)
        predict_input = torch.cat((rnn_output_predict, predict_attn_history, predict_attn_self), dim=2)

        combine_rnn_input = torch.cat((history_input, predict_input), dim=1)
        combine_mask = torch.cat((hm, pm), dim = 1)
        rnn_output_combine = self.combine_rnn(x = combine_rnn_input, x_mask = combine_mask)

        predict_batch_len = pm.size(1)

        ff1_result = F.relu(self.ff1(rnn_output_combine))

        logits = F.sigmoid(self.ff2(ff1_result).squeeze(-1))[:, -predict_batch_len:]

        return logits

    def init_embeddings(self, embedding_matrix):
        self.song_embedding.weight.data.copy_(embedding_matrix)