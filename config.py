import argparse

parser = argparse.ArgumentParser()


parser.add_argument("--max_length",
                    default=10,
                    type=int,
                    help="The maximum input length of (history or predict) tracks")
parser.add_argument("--train_batch_size",
                    default=48,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--eval_batch_size",
                    default=1000,
                    type=int,
                    help="Total batch size for eval.")
parser.add_argument("--learning_rate",
                    default=0.002,
                    type=float,
                    help="The learning rate for Optimizer.")
parser.add_argument("--weight_decay",
                    default=0.0,
                    type=float,
                    help="The weight decay for Optimizer.")
parser.add_argument("--gradient_clipping",
                    default=10.0,
                    type=float,
                    help="The gradient clipping value")
# Common Configs
parser.add_argument("--num_train_epochs",
                    default=50,
                    type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--num_songs",
                    default=3706389,
                    type=int,
                    help="Total number of songs + PAD")
parser.add_argument("--song_embedding_dim",
                    default=30,
                    type=int,
                    help="Dimension of song Embedding")
parser.add_argument("--feature_dim",
                    default=41,
                    type=int,
                    help="Dimension of Feature Embedding for history songs")

# RNN Model Configs
parser.add_argument("--encoder_rnn_hidden_dim",
                    default=100,
                    type=int,
                    help="Dimension of hidden size in encoder rnn")
parser.add_argument("--encoder_rnn_layers",
                    default=1,
                    type=int,
                    help="Number of layers of encoder rnn")
parser.add_argument("--encoder_rnn_dropout",
                    default=0.1,
                    type=float,
                    help="Dropout rate of encoder rnn")
parser.add_argument("--feed_forward_dropout",
                    default=0.2,
                    type=float,
                    help="Dropout rate of fed forward neural network")

# CNN Model Configs
parser.add_argument("--att_num_heads",
                    default=8,
                    type=int,
                    help="Number of heads in Multi Head Attention Model")
parser.add_argument("--cnn_model_dropout",
                    default=0.1,
                    type=float,
                    help="Dropout rate of CNN Model")
parser.add_argument("--cnn_channel_number",
                    default=100,
                    type=int,
                    help="CNN Channel Number")
parser.add_argument("--cnn_connector_dim",
                    default=96,
                    type=int,
                    help="Dimension of CNN connectors")


parser.add_argument("--log_name",
                    default="./log.txt",
                    type=str,
                    help="Name of the log file")
parser.add_argument("--model_name",
                    default='rnnmodel',
                    type=str,
                    help="Model name of saving folder")

main_dir = "/media/data2/Data/wsdm2019/"
# raw csv files
parser.add_argument("--train_set_dir",
                    default=main_dir + "data/training_set",
                    type=str,
                    help="Directory of training raw csv files")
parser.add_argument("--test_set_dir",
                    default=main_dir + "data/test_set",
                    type=str,
                    help="Directory of testing raw csv files")
parser.add_argument("--track_set_dir",
                    default=main_dir + "data/track_features",
                    type=str,
                    help="Directory of track raw csv files")

# post processed python files
parser.add_argument("--train_dir_pkl",
                    default=main_dir + "")


def set_args():
    args = parser.parse_args()
    return args
