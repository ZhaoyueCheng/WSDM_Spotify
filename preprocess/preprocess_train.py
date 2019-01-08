import pandas as pd
import glob
import math
import tqdm

train_set_dir = "/media/data2/Data/wsdm2019/data/training_set/"
test_set_dir = "/media/data2/Data/wsdm2019/data/test_set/"
track_features_dir = "/media/data2/Data/wsdm2019/python/data/track_features/"

# output_dir_csv = "/media/data2/Data/wsdm2019/python/data/training_set/"
output_dir_pkl = "/media/data2/Data/wsdm2019/python/data/train_examples/"

train_files = sorted(glob.glob(train_set_dir+"*.csv"))

bool_columns = ['skip_1', 'skip_2', 'skip_3', 'not_skipped', 'context_switch', 'no_pause_before_play', 'short_pause_before_play',
                'long_pause_before_play', 'hist_user_behavior_is_shuffle', 'premium']
numerical_columns = ['hist_user_behavior_n_seekfwd', 'hist_user_behavior_n_seekback']
numerical_columns_range_map = {'hist_user_behavior_n_seekfwd': (0,100), 'hist_user_behavior_n_seekback': (0,100)}
dropped_columns = ['session_position', 'session_length', 'hour_of_day', 'date'] # temporaryly dropped, will add back if useful
categorical_columns = ['context_type', 'hist_user_behavior_reason_start', 'hist_user_behavior_reason_end']
categorical_columns_map = {'context_type':
                           {'catalog': 0, 'charts': 1, 'editorial_playlist': 2, 'personalized_playlist': 3, 'radio': 4, 'user_collection': 5},
                           'hist_user_behavior_reason_start':
                           {'appload': 0, 'backbtn': 1, 'clickrow': 2, 'endplay': 3, 'fwdbtn': 4, 'playbtn': 5, 'popup': 6, 'remote': 7, 'trackdone': 8, 'trackerror': 9, 'uriopen': 10, 'clickside': 11},
                            'hist_user_behavior_reason_end':
                           {'appload': 0, 'backbtn': 1, 'clickrow': 2, 'endplay': 3, 'fwdbtn': 4, 'logout': 5, 'popup': 6, 'remote': 7, 'trackdone': 8, 'uriopen': 9, 'clickside': 10}
}
id_column = ['session_id', 'track_id_clean']

import pickle
with open (track_features_dir + "track2idx_all.pkl", "rb") as f:
    track2idx = pickle.load(f)

counter = 0

# !temporary! Process last 66 elements
train_files = train_files[66*9:]

for file in train_files:
    print("processing file: " + file)
    df = pd.read_csv(file)

    df_features = df.drop(['session_id', 'track_id_clean'], axis=1)
    new_df = df.copy(deep=True)
    for feature in df_features.columns:
        if feature in bool_columns:
            # for all the bool features, convert to integer [0 --> False, 1--> True]
            new_df[feature] = new_df[feature].astype(int)
        elif feature in numerical_columns:
            # for numerical columns, define min and max for processing
            # do min-max scaling which is (x - min) / (max - min), if value is bigger than max set to max
            (min, max) = numerical_columns_range_map[feature]
            new_df[feature] = new_df[feature].clip(min, max)
            new_df[feature] = (new_df[feature] - min) / (max - min)
        elif feature in categorical_columns:
            # if feature is categorical, convert feature into a map of item --> integer, and leave integer 0 for padding as bool
            #         feature_map, feature_length = convert_feature_to_map(new_df[feature].drop_duplicates().sort_values().values)
            new_df[feature] = new_df[feature].map(categorical_columns_map[feature])
        elif feature in dropped_columns:
            new_df = new_df.drop(feature, axis=1)

    # sanity check
    print("if there is nan: " + str(new_df.isnull().values.any()))

    df_group = new_df.groupby('session_id')
    ngroups = df_group.ngroups

    examples = []
    for name, group in df_group:
        sequence_length = len(group)
        cut_value = math.floor(sequence_length / 2)

        group_history = group.iloc[:cut_value]
        group_predict = group.iloc[cut_value:]

        # group target is the group skip_2 value [0 --> False, 1 --> True]
        group_target = group_predict['skip_2'].values

        # extract tracks data
        tracks_history = [track2idx[a] for a in group_history['track_id_clean'].values]
        tracks_predict = [track2idx[a] for a in group_predict['track_id_clean'].values]

        # extract bool data
        tracks_feature_bool = group_history[bool_columns].values

        # extract numerical data
        tracks_feature_numerical = group_history[numerical_columns].values

        # extract categorical data
        tracks_feature_categorical = group_history[categorical_columns].values

        examples.append({
            'tracks_history': tracks_history,
            'tracks_predict': tracks_predict,
            'tracks_feature_bool': tracks_feature_bool,
            'tracks_feature_numerical': tracks_feature_numerical,
            'tracks_feature_categorical': tracks_feature_categorical,
            'target': group_target
        })

    # sanity check
    print(len(examples))

    import pickle

    with open(output_dir_pkl + file[45:-4] + ".pkl", "wb") as f:
        pickle.dump(examples, f)

    counter += 1

    if counter % 10 == 0:
        print(str(counter / len(train_files)) + " finished")
