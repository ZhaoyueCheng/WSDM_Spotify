import glob
import pandas as pd
import math

train_set_dir = "/media/data2/Data/wsdm2019/data/training_set/"
second_stage_file_dir = "/media/data2/Data/wsdm2019/python/data/second_stage/"

second_stage_files = sorted(glob.glob(second_stage_file_dir+"*.csv"))

def evaluate(submission, groundtruth):
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

for file in second_stage_files:
    print("evaluating file: " + file[52:-17])

    original_file = train_set_dir + file[52:-17] + ".csv"

    train_f = pd.read_csv(original_file)
    logit_f = pd.read_csv(file)

    train_f['logits'] = logit_f['logits']
    test_f = train_f[['session_id', 'skip_2', 'logits']]

    test_groups = test_f.groupby('session_id')

    predicted = []
    groundtruth = []
    for name, group in test_groups:
        cut_value = math.floor(len(group) / 2)
        group_logits = (group['logits'].values[cut_value:] > 0.5).astype(int).tolist()
        group_skip_2 = group['skip_2'].values[cut_value:].tolist()

        groundtruth.append(group_skip_2)
        predicted.append(group_logits)

    print(evaluate(predicted, groundtruth))