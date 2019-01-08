import glob
import numpy as np

ensembe1_dir = "/home/joey/Desktop/ensemble/RNNModelAtt1_Double_LSTM_6294/logits/"
ensembe2_dir = "/home/joey/Desktop/ensemble/RNNModelAtt1_Double_LSTM_6302/logits/"
ensembe3_dir = "/home/joey/Desktop/ensemble/RNNModelAtt1SeperateEncoder_6306/logits/"
ensembe4_dir = "/home/joey/Desktop/ensemble/RNNModelAtt1SeperateEncoder_6309/logits/"
ensembe5_dir = "/home/joey/Desktop/ensemble/RNNModelAtt1SeperateEncoder_6315/logits/"
ensembe6_dir = "/home/joey/Desktop/ensemble/RNNModelAtt1SeperateEncoder_Double_LSTM_6285/logits/"
ensembe7_dir = "/home/joey/Desktop/ensemble/RNNModelAtt1SeperateEncoder_Double_LSTM_6298/logits/"
ensembe8_dir = "/home/joey/Desktop/ensemble/RNNModelAtt1SeperateEncoder_lb_633/logits/"
# ensembe6_dir = "/home/joey/Desktop/ensemble/RNNModelAtt1SeperateEncoder_6309/logits/"

# ensembe1_dir = "/home/joey/Desktop/ensemble/RNNModelAtt1_Double_LSTM_6294/second_stage/same_length/"
# ensembe2_dir = "/home/joey/Desktop/ensemble/RNNModelAtt1SeperateEncoder_6306/second_stage/same_length/"
# ensembe3_dir = "/home/joey/Desktop/ensemble/RNNModelAtt1SeperateEncoder_6315/second_stage/same_length/"
# ensembe4_dir = "/home/joey/Desktop/ensemble/RNNModelAtt1SeperateEncoder_Double_LSTM_6285/second_stage/same_length/"
# ensembe5_dir = "/home/joey/Desktop/ensemble/RNNModelAtt1SeperateEncoder_lb_633/second_stage/same_length/"
# ensembe6_dir = "/home/joey/Desktop/ensemble/RNNModelAtt1SeperateEncoder_6309/second_stage/same_length/"

logit1_files = sorted(glob.glob(ensembe1_dir+"*.csv"))
logit2_files = sorted(glob.glob(ensembe2_dir+"*.csv"))
logit3_files = sorted(glob.glob(ensembe3_dir+"*.csv"))
logit4_files = sorted(glob.glob(ensembe4_dir+"*.csv"))
logit5_files = sorted(glob.glob(ensembe5_dir+"*.csv"))
logit6_files = sorted(glob.glob(ensembe6_dir+"*.csv"))
logit7_files = sorted(glob.glob(ensembe7_dir+"*.csv"))
logit8_files = sorted(glob.glob(ensembe8_dir+"*.csv"))

s = ''

# import pickle
# reg = pickle.load(open('linear_reg', 'rb'))

for index, file in enumerate(logit1_files):
    print(index)
    logit1_f = file
    logit2_f = logit2_files[index]
    logit3_f = logit3_files[index]
    logit4_f = logit4_files[index]
    logit5_f = logit5_files[index]
    logit6_f = logit6_files[index]
    logit7_f = logit7_files[index]
    logit8_f = logit8_files[index]

    f1 = open(logit1_f, "r")
    f2 = open(logit2_f, "r")
    f3 = open(logit3_f, "r")
    f4 = open(logit4_f, "r")
    f5 = open(logit5_f, "r")
    f6 = open(logit6_f, "r")
    f7 = open(logit7_f, "r")
    f8 = open(logit8_f, "r")

    line1 = f1.readline()
    line2 = f2.readline()
    line3 = f3.readline()
    line4 = f4.readline()
    line5 = f5.readline()
    line6 = f6.readline()
    line7 = f7.readline()
    line8 = f8.readline()

    while line1:
        logit1 = line1.split(",")
        logit2 = line2.split(",")
        logit3 = line3.split(",")
        logit4 = line4.split(",")
        logit5 = line5.split(",")
        logit6 = line6.split(",")
        logit7 = line7.split(",")
        logit8 = line8.split(",")

        line1 = f1.readline()
        line2 = f2.readline()
        line3 = f3.readline()
        line4 = f4.readline()
        line5 = f5.readline()
        line6 = f6.readline()
        line7 = f7.readline()
        line8 = f8.readline()

        for i in range(len(logit1)):
            # ensemble_logit = ((float(logit1[i]) + float(logit2[i]) + float(logit3[i]) + float(logit4[i]) + float(logit5[i])) / 5.0)
            # ensemble_logit = ( (float(logit1[i]) + float(logit2[i]) + float(logit3[i]) + float(logit4[i]) + float(
            #     logit5[i]) + float(logit6[i]) ) / 6.0)
            ensemble_logit = ((float(logit1[i]) + float(logit2[i]) + float(logit3[i]) + float(logit4[i]) +
                               float(logit5[i]) + float(logit6[i]) + float(logit7[i]) + float(logit8[i])) / 8.0)
            # ensemble_logit = reg.predict(np.array([[float(logit1[i]), float(logit2[i]), float(logit3[i]), float(logit4[i]), float(logit5[i]), float(logit6[i])]]))[0]
            if ensemble_logit > 0.5:
                s += '1'
            else:
                s += '0'
        s += '\n'

with open("pred.txt", "w") as myfile:
    myfile.write(s)