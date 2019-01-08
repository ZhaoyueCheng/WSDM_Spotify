import glob

ensembe1_dir = "/home/joey/Desktop/ensemble/RNNModelAtt1_Double_LSTM_6302/second_stage/same_length/"
ensembe2_dir = "/home/joey/Desktop/ensemble/RNNModelAtt1SeperateEncoder_6306/second_stage/same_length/"
ensembe3_dir = "/home/joey/Desktop/ensemble/RNNModelAtt1SeperateEncoder_6309/second_stage/same_length/"
ensembe4_dir = "/home/joey/Desktop/ensemble/RNNModelAtt1SeperateEncoder_Double_LSTM_6298/second_stage/same_length/"
# ensembe5_dir = "/home/joey/Desktop/ensemble/RNNModelAtt1SeperateEncoder_lb_633/second_stage/same_length/"
# ensembe6_dir = "/home/joey/Desktop/ensemble/RNNModelAtt1SeperateEncoder_6309/second_stage/same_length/"

target_dir =  "/home/joey/Desktop/ensemble/second_stage_target/"

logit1_files = sorted(glob.glob(ensembe1_dir+"*.csv"))
logit2_files = sorted(glob.glob(ensembe2_dir+"*.csv"))
logit3_files = sorted(glob.glob(ensembe3_dir+"*.csv"))
logit4_files = sorted(glob.glob(ensembe4_dir+"*.csv"))
# logit5_files = sorted(glob.glob(ensembe5_dir+"*.csv"))
# logit6_files = sorted(glob.glob(ensembe6_dir+"*.csv"))

target_files = sorted(glob.glob(target_dir+"*.csv"))

# s = ''

train = []
train_label = []
# valid = []
# valid_label = []

for index, file in enumerate(logit1_files[60:]):
    print(file)
    index += 60
    print(index)
    logit1_f = file
    logit2_f = logit2_files[index]
    logit3_f = logit3_files[index]
    logit4_f = logit4_files[index]
    # logit5_f = logit5_files[index]
    # logit6_f = logit6_files[index]

    target_f = target_files[index]

    f1 = open(logit1_f, "r")
    f2 = open(logit2_f, "r")
    f3 = open(logit3_f, "r")
    f4 = open(logit4_f, "r")
    # f5 = open(logit5_f, "r")
    # f6 = open(logit6_f, "r")

    ft = open(target_f, "r")

    # skip header
    line1 = f1.readline()
    line2 = f2.readline()
    line3 = f3.readline()
    line4 = f4.readline()
    # line5 = f5.readline()
    # line6 = f6.readline()

    linet = ft.readline()

    # start with first line
    line1 = f1.readline()
    line2 = f2.readline()
    line3 = f3.readline()
    line4 = f4.readline()
    # line5 = f5.readline()
    # line6 = f6.readline()

    linet = ft.readline()

    while line1:

        if line1 == "0\n":
            # skip all lines that are history
            line1 = f1.readline()
            line2 = f2.readline()
            line3 = f3.readline()
            line4 = f4.readline()
            # line5 = f5.readline()
            # line6 = f6.readline()

            linet = ft.readline()

            continue


        logit1 = float(line1)
        logit2 = float(line2)
        logit3 = float(line3)
        logit4 = float(line4)
        # logit5 = float(line5)
        # logit6 = float(line6)

        target = int(linet.split(",")[1:][0] == 'True')
        position = int(linet.split(",")[1:][1])

        train.append([logit1, logit2, logit3, logit4, position])
        train_label.append(target)

        line1 = f1.readline()
        line2 = f2.readline()
        line3 = f3.readline()
        line4 = f4.readline()
        # line5 = f5.readline()
        # line6 = f6.readline()

        linet = ft.readline()

import numpy as np
train = np.array(train)
train_label = np.array(train_label)

import pickle
# with open("blend_valid", "wb") as myfile:
# with open("blend_train", "w") as myfile:
#     pickle.dump(train, myfile, protocol=4)

# with open("blend_valid_target", "wb") as myfile:
# with open("blend_train_target", "w") as myfile:
#     pickle.dump(train_label, myfile, protocol=4)

# np.save("blend_train", train)
# np.save("blend_train_target", train_label)

np.save("blend_valid", train)
np.save("blend_valid_target", train_label)