import os
from os import listdir
from os.path import isfile, join
import csv
import tqdm
import copy
import numpy as np
import pandas as pd


def mapbool(s):
    return s.lower in ['true', '1']

train_set_dir = "/media/data1/Data/wsdm2019/data/training_set/"
test_set_dir = "/media/data1/Data/wsdm2019/data/test_set/"
track_features_dir = "/media/data1/Data/wsdm2019/data/track_features/"

# output_dir = "/media/data1/Data/wsdm2019/data/python/"

track_file_list = sorted(os.listdir(track_features_dir))

track_feature_index = ['track_id', 'duration', 'release_year', 'us_popularity_estimate', 'acousticness', 'beat_strength', 'bounciness', 'danceability', 'dyn_range_mean', 'energy', 'flatness', 'instrumentalness', 'key', 'liveness', 'loudness', 'mechanism', 'mode', 'organism', 'speechiness', 'tempo', 'time_signature', 'valence', 'acoustic_vector_0', 'acoustic_vector_1', 'acoustic_vector_2', 'acoustic_vector_3', 'acoustic_vector_4', 'acoustic_vector_5', 'acoustic_vector_6', 'acoustic_vector_7']

session_id_map = {}
session_length_map = {}
track_id_map = {}

session_id_counter = 0
track_id_counter = 0

# for file_name in tqdm(file_list):
c = 0

for file_name in track_file_list:
    file_dir = track_features_dir + file_name

    examples = []

    with open(file_dir, 'r') as file:
        f = csv.reader(file)

        # skip header
        for row in list(f)[1:]:
            example = copy.deepcopy(row)[-7:]

            examples.append(example)

            c = c + 1
            if c % 1000 == 0:
                print(c)




acoustic_vector = np.array(examples, dtype=float)

print('Done')