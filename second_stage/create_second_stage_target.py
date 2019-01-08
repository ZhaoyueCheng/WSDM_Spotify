import pandas as pd
import glob
import math
import tqdm

train_set_dir = "/media/data2/Data/wsdm2019/data/training_set/"
test_set_dir = "/media/data2/Data/wsdm2019/data/test_set/"
track_features_dir = "/media/data2/Data/wsdm2019/python/data/track_features/"

# output_dir_csv = "/media/data2/Data/wsdm2019/python/data/training_set/"
output_dir = "/home/joey/Desktop/ensemble/second_stage_target/"

train_files = sorted(glob.glob(train_set_dir+"*.csv"))

# Second Stage is on last 66 files
train_files = train_files[66*9:]

for file in train_files:
    print("processing file: " + file)
    df = pd.read_csv(file)

    new_df = df.copy(deep=True)[['skip_2', 'session_position']]
    new_df.to_csv(output_dir + file[45:], header="Target")