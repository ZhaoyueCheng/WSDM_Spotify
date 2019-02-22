import pandas as pd
import glob

train_set_dir = "/media/data2/Data/wsdm2019/data/training_set/"
test_set_dir = "/media/data2/Data/wsdm2019/data/test_set/"
track_features_dir = "/media/data2/Data/wsdm2019/python/data/track_features/"

output_dir = "/media/data2/Data/wsdm2019/python/data/training_set/"

train_files = sorted(glob.glob(train_set_dir+"*.csv"))

num_combine= 66
files_chunk = []
for i in range(0, len(train_files), num_combine):
    files_chunk.append(train_files[i: i+ num_combine])

for chunk, files in enumerate(files_chunk):
    print("chunk " + str(chunk))
    list_ = []
    for file in files:
        df = pd.read_csv(file)
        list_.append(df)
    frame = pd.concat(list_, axis=0, ignore_index=True)
    frame.to_csv(output_dir+"log_"+str(chunk))