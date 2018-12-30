from config import set_args
from models.rnn_model import *
from models.layers import *
import torch
import numpy as np
import math
import copy
import pickle
import random
import datetime
import tqdm
from tqdm import trange
import shutil
import glob

args = set_args()

train_set_dir = "/media/data2/Data/wsdm2019/data/training_set/"
test_set_dir = "/media/data2/Data/wsdm2019/data/test_set/"
track_features_dir = "/media/data2/Data/wsdm2019/python/data/track_features/"

train_files = "/media/data2/Data/wsdm2019/python/data/train_examples/"
train_dir_pkl = "/media/data2/Data/wsdm2019/python/data/train_examples/"
model_dir = "/media/data2/Data/wsdm2019/python/model/rnnattn1/"
# model_dir = "/media/data2/Data/wsdm2019/python/model/rnnmodel/"

second_stage_file_dir = "/media/data2/Data/wsdm2019/python/data/second_stage/"

categorical_feature_max_len = [6, 12, 11]

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form for categorical features

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = np.eye(num_classes)
    return y[labels]

# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, model_dir + 'model_best')

def batchGen(examples):
    history_tracks = []
    history_features = []
    history_sequence_masks = []

    predict_tracks = []
    targets = []
    predict_sequence_masks = []

    # loss_masks = []
    # sequence_masks = []

    tracks_history = [example['tracks_history'] for example in examples]
    tracks_predict = [example['tracks_predict'] for example in examples]
    tracks_feature_bool = [example['tracks_feature_bool'] for example in examples]
    tracks_feature_numerical = [example['tracks_feature_numerical'] for example in examples]
    tracks_feature_categorical = [example['tracks_feature_categorical'].astype(int) for example in examples]
    target_original = [example['target'].tolist() for example in examples]

    history_lengths = [len(track) for track in tracks_history]
    predict_lengths = [len(track) for track in tracks_predict]

    for trackh, boolf, numf, catf, lengthh in zip(tracks_history, tracks_feature_bool, tracks_feature_numerical, tracks_feature_categorical, history_lengths):
        pad_length = max_length - lengthh

        # create history song inputs
        history_track = trackh + [0] * pad_length
        history_tracks.append(history_track)

        # create skip inputs and mask out the correct answer
        history_feature = np.concatenate((boolf, numf), axis=1)
        for i in range(len(categorical_feature_max_len)):
            history_feature = np.concatenate((history_feature, one_hot_embedding(catf[:,i], categorical_feature_max_len[i])), axis=1)
        history_feature = np.concatenate((history_feature, np.zeros([pad_length, args.feature_dim])), axis=0)
        history_features.append(history_feature)
        #
        # # create targets
        # target = copy.deepcopy(skip2)
        # target = [item - 2 for item in target] + [0] * pad_length
        # targets.append(target)
        #
        # # create loss masks
        # mask = [0.0] * cut + [1.0] * target_length + [0.0] * pad_length
        # loss_masks.append(mask)
        #
        # # # create weighted loss masks to reward more for songs come in front
        # # loss_weight_mask = [0.0] * cut + [1- (1/target_length)*x for x in range(target_length)] + [0.0] * pad_length
        # # loss_weight_masks.append(loss_weight_mask)
        # # # convert into FloatTensor and do normalize
        # # tensor_weighted_mask = torch.cuda.FloatTensor(loss_weight_masks)
        # # tensor_weighted_mask = tensor_weighted_mask / tensor_weighted_mask.sum(dim=1).unsqueeze(-1)

        # create history sequence masks
        history_sequence_mask = [0] * lengthh + [1] * pad_length
        history_sequence_masks.append(history_sequence_mask)

    for trackp, lengthp, targ in zip(tracks_predict, predict_lengths, target_original):
        pad_length = max_length - lengthp

        # create target song inputs
        predict_track = trackp + [0] * pad_length
        predict_tracks.append(predict_track)

        # create targets
        target = targ + [0] * pad_length
        targets.append(target)

        # create predict sequence masks
        predict_sequence_mask = [0] * lengthp + [1] * pad_length
        predict_sequence_masks.append(predict_sequence_mask)


    return torch.cuda.LongTensor(history_tracks), torch.cuda.FloatTensor(history_features), torch.cuda.ByteTensor(history_sequence_masks), \
           torch.cuda.LongTensor(predict_tracks), torch.cuda.ByteTensor(predict_sequence_masks), torch.cuda.FloatTensor(targets)


def train_model(model, optim, loss_fcn, args, train_examples, eval_examples, epoch, best_acc):
    step = 0

    model.train()

    dataset_size = len(train_examples)
    batch_size = args.train_batch_size

    avg_acc = 0.0
    avg_loss = 0.0

    random.shuffle(train_examples)

    for start_index in trange(0, dataset_size, batch_size):
        end_index = min(start_index + batch_size, dataset_size)
        batch = train_examples[start_index: end_index]

        ht, hf, hm, pt, pm, t = batchGen(batch)

        lm = (1 - pm).float()

        encoder_result = model.forward(ht, hf, hm, pt, pm)

        # 1 - pm because mask is 0 for real value and 1 for padding values
        # loss = (loss_fcn(encoder_result, t) * lm).mean()
        loss = (loss_fcn(encoder_result, t) * lm).sum() / lm.sum()

        lmd = lm.detach().cpu().numpy()

        # calculate acc
        res = (encoder_result.detach().cpu().numpy() > 0.5) * lmd
        acc = ((res == t.detach().cpu().numpy()) * lmd).sum() / lmd.sum()

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)

        optim.step()

        avg_acc += acc
        avg_loss += loss.detach().cpu().numpy()

        # loss_print_step = 1000
        # if step % loss_print_step == 0:
        step += 1

    now = datetime.datetime.now()

    print(now.strftime("%H:%M") + ": TRAIN epoch: " + str(epoch) + " step: " + str(step) +
          " train accuracy: " + str(avg_acc / step) + " loss: " + str(avg_loss / step))
    # avg_acc = 0
    # avg_loss = 0

        # if step % 1800 == 0:
        #     ap = eval_model(model, args, eval_examples, epoch)
        #     model.train()
        #
        #     is_best = False
        #     if ap > best_acc:
        #         best_acc = ap
        #         is_best = True
        #
        #     save_checkpoint({
        #         'epoch': epoch,
        #         'state_dict': model.state_dict(),
        #         'optimizer': optim.state_dict(),
        #         'current_acc': ap,
        #     }, is_best, filename=model_dir+'epoch_'+str(epoch))

        # step += 1

def eval_model(model, optim, loss_fcn, args, eval_examples, epoch):

    # for official evaluating script
    groundtruth = []
    predicted = []

    model.eval()
    batch_size = args.eval_batch_size
    dataset_size = len(eval_examples)
    avg_acc = 0
    avg_loss = 0
    step = 0

    for start_index in range(0, dataset_size, batch_size):
        end_index = min(start_index + batch_size, dataset_size)
        batch = eval_examples[start_index: end_index]

        ht, hf, hm, pt, pm, t= batchGen(batch)

        lm = (1 - pm).float()

        encoder_result = model.forward(ht, hf, hm, pt, pm)

        # 1 - pm because mask is 0 for real value and 1 for padding values
        # loss = (loss_fcn(encoder_result, t) * lm).mean()
        loss = (loss_fcn(encoder_result, t) * lm).sum() / lm.sum()

        lmd = lm.detach().cpu().numpy()

        res = (encoder_result.detach().cpu().numpy() > 0.5) * lmd
        acc = ((res == t.detach().cpu().numpy()) * lmd).sum() / lmd.sum()

        gt = t.detach().cpu().numpy()

        for (x, y, z) in zip(res, gt, lmd):
            nonzero = z.nonzero()
            groundtruth.append(y[nonzero])
            predicted.append(x[nonzero])

        step += 1
        avg_acc += acc
        avg_loss += loss.detach().cpu().numpy()

    ap, first_pred_acc = evaluate(predicted, groundtruth)

    print("EVAL ap: " + str(ap) + " first_pred_acc: " + str(first_pred_acc))
    print("EVAL epoch: " + str(epoch) + " test accuracy: " + str(avg_acc / step) +
          " loss: " + str(avg_loss / step))

    return ap

def inference_model(model, optim, args, inf_examples, inf_file_name):

    new_file_name = second_stage_file_dir + inf_file_name[54:-4] + "_second_stage.csv"

    print(new_file_name)

    with open(new_file_name, "w+") as myfile:
        myfile.write("logits\n")

    model.eval()
    batch_size = args.eval_batch_size
    dataset_size = len(inf_examples)

    for start_index in trange(0, dataset_size, batch_size):
        end_index = min(start_index + batch_size, dataset_size)
        batch = inf_examples[start_index: end_index]

        ht, hf, hm, pt, pm, t = batchGen(batch)

        lm = (1 - pm).float()

        encoder_result = model.forward(ht, hf, hm, pt, pm)

        lmd = lm.detach().cpu().numpy()

        logits = encoder_result.detach().cpu().numpy()

        # res = (encoder_result.detach().cpu().numpy() > 0.5) * lmd

        # Start building string
        s = ''

        historylen = hm.eq(0).detach().cpu().numpy()

        for (x, y, z) in zip(logits, historylen, lmd):
            nonzero = z.nonzero()
            logit = x[nonzero]

            num_zero = np.sum(y)

            s += '0\n' * num_zero

            for i in logit:
                s += str(i)
                s += '\n'

        with open(new_file_name, "a") as myfile:
            myfile.write(s)


def evaluate(submission,groundtruth):
    ap_sum = 0.0
    first_pred_acc_sum = 0.0
    counter = 0
    for sub, tru in zip(submission, groundtruth):
        if len(sub) != len(tru):
            raise Exception('Line {} should contain {} predictions, but instead contains '
                            '{}'.format(counter+1,len(tru),len(sub)))
        ap_sum += ave_pre(sub,tru,counter)
        first_pred_acc_sum += sub[0] == tru[0]
        counter+=1
    ap = ap_sum/counter
    first_pred_acc = first_pred_acc_sum/counter
    return ap, first_pred_acc

def ave_pre(submission,groundtruth,counter):
    s = 0.0
    t = 0.0
    c = 1.0
    for x, y in zip(submission, groundtruth):
        if x != 0 and x != 1:
            raise Exception('Invalid prediction in line {}, should be 0 or 1'.format(counter))
        if x==y:
            s += 1.0
            t += s / c
        c += 1
    return t/len(groundtruth)

# get all the maps and vectors
with open(track_features_dir + "music_vector_all.pkl", "rb") as f:
    vector = pickle.load(f)
with open(track_features_dir + "track2idx_all.pkl", "rb") as f:
    track2idx = pickle.load(f)

# with open("/media/data2/Data/wsdm2019/python/data/sample_examples", "rb") as f:
#     train_examples_load = pickle.load(f)
# with open(train_dir_pkl+"temp_valid_examples", "rb") as f:
#     valid_examples = pickle.load(f)

pkl_files = sorted(glob.glob(train_dir_pkl+"*.pkl"))
train_pkl_files = pkl_files[:600]
    #.extend(pkl_files[42:])
# valid_pkl_files = pkl_files[400:402]


valid_pkl_files = pkl_files[600:605]

inference_pkl_files = pkl_files[-66:]

valid_examples = []
for valid_pkl_file in valid_pkl_files:
    with open(valid_pkl_file, "rb") as f:
        valid_examples.extend(pickle.load(f))

# random.seed(10)
# random.shuffle(train_examples_load)
# train_examples = train_examples_load[:int(len(train_examples_load) * 0.75)]
# valid_examples = train_examples_load[int(len(train_examples_load) * 0.75):]

music_embedding = torch.Tensor(vector)

# brute force pad everything to length 10 since all the (history/predict) can have maximum of 10 songs
max_length = args.max_length

# model = RNNModel(args)
model = RNNModelAtt1(args)
model.cuda()
model.init_embeddings(music_embedding)

optimizer = torch.optim.Adamax(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)

loss_fcn = torch.nn.BCELoss(reduction='none')

best_acc = 0

# load checkpoint
checkpoint = torch.load(model_dir + 'model_best')
print('epoch: ' + str(checkpoint['epoch']))
print('acc: ' + str(checkpoint['current_acc']))
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])


best_acc = 0

print("evaluation after loading")

ap = eval_model(model, optimizer, loss_fcn, args, valid_examples, -1)

for inf_pkl_file in inference_pkl_files:
    with open(inf_pkl_file, "rb") as f:

        inference_examples = pickle.load(f)

        # train_model(model, optimizer, loss_fcn, args, train_examples, valid_examples, epoch, best_acc)

        # if (epoch + 1) % 1 == 0:

        inference_model(model, optimizer, args, inference_examples, inf_pkl_file)



    # print("Epoch " + str(epoch) + " Done.")