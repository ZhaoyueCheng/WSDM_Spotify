import pandas as pd
import glob
import math
import tqdm


def convert_features(new_df):
    df_features = new_df.drop(['session_id', 'track_id_clean'], axis=1)
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

    return new_df


train_set_dir = "/media/data2/Data/wsdm2019/data/training_set/"
test_set_dir = "/media/data2/Data/wsdm2019/data/test_set/"
track_features_dir = "/media/data2/Data/wsdm2019/python/data/track_features/"

# output_dir_csv = "/media/data2/Data/wsdm2019/python/data/training_set/"
output_dir_pkl = "/media/data2/Data/wsdm2019/python/data/test_examples/"

test_files = sorted(glob.glob(test_set_dir+"*.csv"))

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

batch = 10
print("processing batch " + str(batch * 6) + " to " + str((batch+1) * 6))

test_files_history = test_files[66:][batch * 6: (batch+1) *6]
test_files_predict = test_files[:66][batch * 6: (batch+1) *6]

import pickle
with open (track_features_dir + "track2idx_all.pkl", "rb") as f:
    track2idx = pickle.load(f)

counter = 0
for file_history, file_predict in zip(test_files_history, test_files_predict):
    print("processing file history: " + file_history)
    print("processing file predict: " + file_predict)
    df_hist_raw = pd.read_csv(file_history)
    df_predict_raw = pd.read_csv(file_predict)

    df_hist_converted = convert_features(df_hist_raw)
    df_predict_converted = convert_features(df_predict_raw)

    df_group_hist = df_hist_converted.groupby('session_id')
    df_group_predict = df_predict_converted.groupby('session_id')

    ngroups = df_group_hist.ngroups

    examples = []
    for name, group_history in df_group_hist:
        group_predict = df_group_predict.get_group(name)

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
            'tracks_feature_categorical': tracks_feature_categorical
        })

    # sanity check
    print(len(examples))

    import pickle

    with open(output_dir_pkl + file_history[56:-4] + ".pkl", "wb") as f:
        pickle.dump(examples, f)

    counter += 1

    if counter % 2 == 0:
        print(str(counter / len(test_files_history)) + " finished")
