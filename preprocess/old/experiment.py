import os
from os import listdir
from os.path import isfile, join
import csv
import tqdm
import copy
import pandas as pd
import numpy as np
import glob


def mapbool(s):
    return s.lower in ['true','1']

train_set_dir = "/media/data2/Data/wsdm2019/data/training_set/"
test_set_dir = "/media/data1/Data/wsdm2019/data/test_set/"
track_features_dir = "/media/data1/Data/wsdm2019/data/test_set/"

output_dir = "/media/data1/Data/wsdm2019/data/python/"

train_files_dir = sorted(glob.glob(train_set_dir +  "*.csv"))

# large_train_set_dir = "/media/data2/Data/wsdm2019/python/data/training_set/"

# track_feature_files = sorted(glob.glob(track_features_dir+"*.csv"))
#
# # for file_dir in track_feature_files:
# track_feature_dir_0 = track_feature_files[0]
# track_feature_0 = pd.read_csv(track_feature_dir_0)
#
# track_feature_dir_1 = track_feature_files[1]
# track_feature_1 = pd.read_csv(track_feature_dir_1)
#
# track_feature_all = track_feature_0.append(track_feature_1)

# large_files = sorted(glob.glob(large_train_set_dir + "*.csv"))
#
# large_file = large_files[1]
#
df = pd.read_csv(train_files_dir[0])

groups = df.groupby('session_id')

for name, group in groups:
    print(group['context_type'].drop_duplicates().shape[0])

# print(df['hist_user_behavior_n_seekfwd'].max())
# print(df['hist_user_behavior_n_seekback'].max())
# print(df['hour_of_day'].max())
#
# print(df['hist_user_behavior_n_seekfwd'].min())
# print(df['hist_user_behavior_n_seekback'].min())
# print(df['hour_of_day'].min())
#
# print(df['hist_user_behavior_reason_start'].drop_duplicates().sort_values())
# print(df['hist_user_behavior_reason_end'].drop_duplicates().sort_values())
# print(df['context_type'].drop_duplicates().sort_values())
# file_list = sorted(os.listdir(train_set_dir))
#
# train_feature_index = ['session_id', 'session_position', 'session_length', 'track_id_clean', 'skip_1', 'skip_2', 'skip_3', 'not_skipped', 'context_switch', 'no_pause_before_play', 'short_pause_before_play', 'long_pause_before_play', 'hist_user_behavior_n_seekfwd', 'hist_user_behavior_n_seekback', 'hist_user_behavior_is_shuffle', 'hour_of_day', 'date', 'premium', 'context_type', 'hist_user_behavior_reason_start', 'hist_user_behavior_reason_end']
#
# session_id_map = {}
# session_length_map = {}
# track_id_map = {}
#
# session_id_counter = 0
# track_id_counter = 0
#
# behavior = []
# start = []
# end = []
#
# for file_name in tqdm.tqdm(file_list):
# # for file_name in file_list:
#     file_dir = train_set_dir + file_name
#
#     examples = []
#
#     with open(file_dir, 'r') as file:
#         f = csv.reader(file)
#
#         # skip header
#         for row in list(f)[1:]:
#             # example = copy.deepcopy(row)
#             example = row
#
#             # # feature 0 session_id
#             # if example[0] in session_id_map.keys():
#             #     example[0] = session_id_map[example[0]]
#             # else:
#             #     session_id_map[example[0]] = session_id_counter
#             #     example[0] = session_id_counter
#             #     session_id_counter += 1
#             #
#             # # feature 2 session position
#             # example[1] = int[example[1]]
#             #
#             # # feature 3 session length
#             # example[2] = int[example[2]]
#             # if example[2] not in session_length_map.keys():
#             #     session_length_map[example[0]] = example[2]
#             #
#             # # feature 4 track_id_clean
#             # if example[3] in track_id_map.keys():
#             #     example[3] = track_id_map[example[3]]
#             # else:
#             #     track_id_map[example[3]] = track_id_counter
#             #     example[3] = track_id_counter
#             #     track_id_counter += 1
#
#             # # bool_features
#             # bool_feature_indexes = [4,5,6,7,8,9,10,11,14,17]
#             # for i in bool_feature_indexes:
#             #     example[i] = mapbool
#
#             if example[-3] not in behavior:
#                 behavior.append(example[-3])
#
#             if example[-2] not in start:
#                 start.append(example[-2])
#
#             if example[-1] not in end:
#                 end.append(example[-1])
#
#         print(behavior)
#         print(start)
#         print(end)
#
# print('Done')
