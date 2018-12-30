import glob
import csv
import tqdm
import copy
import numpy as np
import pandas as pd
import catboost
import math

# Load tracks

track_features_dir = "/media/data2/Data/wsdm2019/data/track_features/"
track_feature_files = sorted(glob.glob(track_features_dir+"*.csv"))
track_feature_dir_0 = track_feature_files[0]
track_feature_0 = pd.read_csv(track_feature_dir_0)

track_feature_dir_1 = track_feature_files[1]
track_feature_1 = pd.read_csv(track_feature_dir_1)

track_feature_all = track_feature_0.append(track_feature_1)

# get this so that it's easier to merge with train indexes
track_feature_id_index = track_feature_all.set_index('track_id')


# # Load session features into dataframe

train_set_dir = "/media/data2/Data/wsdm2019/data/training_set/"
train_files = sorted(glob.glob(train_set_dir+"*.csv"))

catboost_files = train_files[500:]

temp_catboost_train_file_size = 5
temp_catboost_valid_file_size = 1

catboost_train_files = catboost_files[:temp_catboost_train_file_size]
catboost_valid_files = catboost_files[temp_catboost_train_file_size:temp_catboost_valid_file_size + temp_catboost_train_file_size]


train_df_list = []
valid_df_list = []

for f in catboost_train_files:
    train_df_list.append(pd.read_csv(f))    

for f in catboost_valid_files:
    valid_df_list.append(pd.read_csv(f))
    
catboost_train_raw = pd.concat(train_df_list, ignore_index=True)
catboost_valid_raw = pd.concat(valid_df_list, ignore_index=True)


catboost_train_raw_group = catboost_train_raw.groupby('session_id')
ngroups = catboost_train_raw_group.ngroups


# for name, group in df_group:
    

# catboost_train_raw_group_history = catboost_train_raw.groupby('session_id').apply(lambda x: x.iloc[:math.floor(x['session_position'].size / 2)])
# catboost_train_raw_group_predict = catboost_train_raw.groupby('session_id').apply(lambda x: x.iloc[math.floor(x['session_position'].size / 2):])

# catboost_train_raw_group_history.get_group('38_00012a3f-dff9-4560-b18b-103cfecf8a43')

# catboost_train_raw_group_predict.get_group('38_00012a3f-dff9-4560-b18b-103cfecf8a43')


# In[67]:


test_group = catboost_train_raw_group.get_group('38_00012a3f-dff9-4560-b18b-103cfecf8a43')


# In[77]:


test_group.head()


# In[73]:


test_group_history = test_group.iloc[:math.floor(test_group['session_position'].size / 2)].rename(index=str, columns={"track_id_clean": "track_id"})
test_group_predict = test_group.iloc[math.floor(test_group['session_position'].size / 2):].rename(index=str, columns={"track_id_clean": "track_id"})


test_group_target = test_group_predict.iloc[:, 5]

test_group_predict = test_group_predict.iloc[:,:4]
test_group_predict.head()


session_dropped_columns = ['session_id', 'track_id', 'session_position']
session_bool_columns = ['skip_1', 'skip_2', 'skip_3', 'not_skipped', 'context_switch', 'no_pause_before_play', 'short_pause_before_play',
                'long_pause_before_play', 'hist_user_behavior_is_shuffle', 'premium']
session_numerical_columns = ['hist_user_behavior_n_seekfwd', 'hist_user_behavior_n_seekback']
session_numerical_columns_range_map = {'hist_user_behavior_n_seekfwd': (0,100), 'hist_user_behavior_n_seekback': (0,100)}
session_categorical_columns = ['context_type', 'hist_user_behavior_reason_start', 'hist_user_behavior_reason_end', 'session_length', 'hour_of_day', 'date']


track_dropped_columns = ['track_id']
track_numerical_columns = ['duration', 'us_popularity_estimate', 'acousticness', 'beat_strength', 'bounciness', 'danceability',
                             'dyn_range_mean', 'energy', 'flatness', 'instrumentalness', 'liveness', 'loudness', 'mechanism', 
                             'organism', 'speechiness', 'tempo', 'valence']
track_vector_columns = ['acoustic_vector_0', 'acoustic_vector_1', 'acoustic_vector_2', 
                        'acoustic_vector_3', 'acoustic_vector_4', 'acoustic_vector_5',
                        'acoustic_vector_6', 'acoustic_vector_7']
track_categorical_columns = ['release_year', 'key', 'mode', 'time_signature']


# In[308]:


# def get_session_history_summarization(test_group_history):
#     num_feature_list = []
#     cat_feature_list = []
#     for feature in test_group_history.columns:
#         if feature in history_bool_columns:
#             col = test_group_history[feature].astype(int)
#             num_feature_list.extend([col.mean(), col.var(), col.mode()[0]])
#         elif feature in history_numerical_columns:
#             (mi, ma) = history_numerical_columns_range_map[feature]
#             col = test_group_history[feature].clip(mi, ma)
#             col = (col - mi) / (ma - mi)
#             num_feature_list.extend([col.mean(), col.var()])
#         elif feature in history_categorical_columns:
#             col = test_group_history[feature]
#             cat_feature_list.append(col.mode()[0])
#     return num_feature_list, cat_feature_list
def get_session_history_summarization(test_group_history):
    num_feature_list = []
    cat_feature_list = []
    
    for feature in test_group_history.columns:
        if feature in session_numerical_columns:
            (mi, ma) = session_numerical_columns_range_map[feature]
            col = test_group_history[feature].clip(mi, ma)
            col = (col - mi) / (ma - mi)
            num_feature_list.extend([col.mean(), col.var()])
    
    bool_cols = test_group_history[session_bool_columns].astype(int)
    num_feature_list.extend(bool_cols.mean().tolist())
    num_feature_list.extend(bool_cols.var().tolist())
    num_feature_list.extend(bool_cols.mode().iloc[0].tolist())
    
    cat_cols = test_group_history[session_categorical_columns]
    
    cat_feature_list.extend(cat_cols.mode().iloc[0].tolist())
    
    return num_feature_list, cat_feature_list


def get_track_df(track_list):
    return track_feature_id_index.loc[track_list]


def get_current_track_feature(self_track_df):
    num_feature_list = []
    cat_feature_list = []
#     for feature in self_track_df.columns:
#         print(feature)
#     if feature in predict_numerical_columns:
    num_feature_list.extend(self_track_df[track_numerical_columns].tolist())
    cat_feature_list.extend(self_track_df[track_categorical_columns].tolist())
    return num_feature_list, cat_feature_list


def get_tracks_df_summarization_feature(tracks_df):
    num_feature_list = []
    cat_feature_list = []
    
    num_feature_list.extend(tracks_df[track_numerical_columns]).mean()
    num_feature_list.extend(tracks_df[track_numerical_columns]).var()
    
    cat_feature_list.extend(tracks_df[track_categorical_columns].mode().iloc[0].tolist())
    
    return num_feature_list, cat_feature_list


# first get the summarization data
num_feat_hist, cat_feat_hist = get_session_history_summarization(test_group_history)

for index, row in test_group_predict.iterrows():
    num_feat_track = []
    cat_feat_track = []
    session_position = row[1]
    print(index, row[1])
#     process_session(index, test_group_history, test_group_predict test_group_history_tracks)
    break;

