from models.layers_cnn import *
from models.layers import *

class CNNModel(nn.Module):
    def __init__(self, args):
        super(CNNModel, self).__init__()
        self.song_embedding = nn.Embedding(num_embeddings=args.num_songs, embedding_dim=args.song_embedding_dim,
                                           padding_idx=0)

        self.history_embed = Embedding((args.song_embedding_dim + args.feature_dim), args.cnn_channel_number)
        self.predict_embed = Embedding(args.song_embedding_dim, args.cnn_channel_number)

        self.history_encoder = EncoderBlock(conv_num=4, ch_num=args.cnn_channel_number, k=3, args=args)
        self.predict_encoder = EncoderBlock(conv_num=4, ch_num=args.cnn_channel_number, k=3, args=args)

        self.model_enc_blks = nn.ModuleList([EncoderBlock(conv_num=2, ch_num=args.cnn_channel_number, k=3, args=args) for _ in range(7)])

        self.ff1 = FeedForwardNetwork(input_size=args.cnn_channel_number,
                                      hidden_size=int(args.cnn_channel_number / 2),
                                      output_size=1, dropout_rate=args.cnn_model_dropout)

        self.dropout_rate = args.cnn_model_dropout


    def forward(self, ht, hf, hm, pt, pm):

        # Encode history song and feature, predict song seperately
        song_vec_history = self.song_embedding(ht)
        song_vec_predict = self.song_embedding(pt)
        history_input = torch.cat((song_vec_history, hf), dim=2)

        history_emb = self.history_embed(history_input)
        predict_emb = self.predict_embed(song_vec_predict)

        history_enc = self.history_encoder(history_emb, hm, 1, 1)
        predict_enc = self.predict_encoder(predict_emb, pm, 1, 1)

        combine_mask = torch.cat((hm, pm), dim=1)

        combine_enc = torch.cat((history_enc, predict_enc), dim=2)
        enc_in = F.dropout(combine_enc, p=self.dropout_rate, training=self.training)

        for i, blk in enumerate(self.model_enc_blks):
            enc_in = blk(enc_in, combine_mask, i*(2+2) + 1, 7)

        out = enc_in.transpose(1, 2)

        ff1_result = self.ff1(out)

        predict_batch_len = pm.size(1)

        logits = F.sigmoid(ff1_result)[:, -predict_batch_len:].squeeze(-1)

        return logits


    def init_embeddings(self, embedding_matrix):
        self.song_embedding.weight.data.copy_(embedding_matrix)


