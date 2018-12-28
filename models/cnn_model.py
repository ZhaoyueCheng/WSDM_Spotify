from models.layers_cnn import *

class CNNModel(nn.Module):
    def __init__(self, args):
        super(CNNModel, self).__init__()

    def forward(self, ht, hf, hm, pt, pm):
        self.song_embedding = nn.Embedding(num_embeddings=args.num_songs, embedding_dim=args.song_embedding_dim, padding_idx=0)

        self.history_enc = EncoderBlock(conv_num=4, ch_num=args., )

    def init_embeddings(self, embedding_matrix):
        self.song_embedding.weight.data.copy_(embedding_matrix)

