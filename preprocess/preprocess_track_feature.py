
# coding: utf-8

# In[4]:


import pandas as pd
import glob


# In[5]:


train_set_dir = "/media/data2/Data/wsdm2019/data/training_set/"
test_set_dir = "/media/data2/Data/wsdm2019/data/test_set/"
track_features_dir = "/media/data2/Data/wsdm2019/data/track_features/"

output_dir = "/media/data2/Data/wsdm2019/python/data/track_features/"


# In[6]:


track_feature_files = sorted(glob.glob(track_features_dir+"*.csv"))


# In[7]:


# for file_dir in track_feature_files:
track_feature_dir_0 = track_feature_files[0]
track_feature_0 = pd.read_csv(track_feature_dir_0)
track_feature_0.head()


# In[8]:


track_feature_dir_1 = track_feature_files[1]
track_feature_1 = pd.read_csv(track_feature_dir_1)
track_feature_1.head()


# In[9]:


track_feature_1.columns


# In[10]:


track_feature_all = track_feature_0.append(track_feature_1)


# In[63]:


id_columns = ['track_id']
normalize_columns = ['duration', 'release_year', 'us_popularity_estimate',
                    'acousticness', 'beat_strength', 'bounciness', 'danceability',
                    'dyn_range_mean', 'energy', 'flatness', 'instrumentalness', 'key',
                    'liveness', 'loudness', 'mechanism', 'organism', 'speechiness',
                    'tempo', 'time_signature', 'valence']
no_normalize_columns = ['acoustic_vector_0', 'acoustic_vector_1', 'acoustic_vector_2', 
                        'acoustic_vector_3', 'acoustic_vector_4', 'acoustic_vector_5',
                        'acoustic_vector_6', 'acoustic_vector_7']
categorical_columns = ['mode']


# In[64]:


track_feature_copy = track_feature_all.copy(deep=True)


# In[65]:


track_feature_copy.head()


# In[66]:


pd.set_option('display.max_columns', None)


# In[67]:


# do min-max scaling which is (x - min) / (max - min), if value is bigger than max set to max
for feature in normalize_columns:
    min = track_feature_copy[feature].min()
    max = track_feature_copy[feature].max()
    track_feature_copy[feature] = track_feature_copy[feature].clip(min, max)
    track_feature_copy[feature] = (track_feature_copy[feature] - min) / (max - min)

# do one hot encoding for the categorical variable
for feature in categorical_columns:
    track_feature_copy = pd.concat([track_feature_copy, pd.get_dummies(track_feature_copy[feature], prefix=feature)], axis=1)
    track_feature_copy = track_feature_copy.drop(feature, axis=1)


# In[68]:


track_feature_copy.head()


# In[69]:


track_feature_copy.describe()


# In[70]:


# track_feature_all.to_pickle(output_dir + "track_feature_all_df.pkl")


# In[71]:


# track_feature_selected.to_pickle(output_dir + "track_feature_selected_df.pkl")


# In[72]:


# insert the padding embedding
track_feature_copy.loc[-1] = ["PAD"]+([0]*(track_feature_copy.shape[1] - 1))
track_feature_copy.index = track_feature_copy.index + 1
track_feature_copy = track_feature_copy.reset_index().drop('index', axis=1)


# In[73]:


track_feature_copy.head()


# In[74]:


track_feature_copy.copy(deep=True)['track_id'].drop_duplicates().shape


# In[75]:


idx2track = track_feature_copy.to_dict()['track_id']
# sanity check
print(len(idx2track))


# In[76]:


track2idx = {v: k for k, v in idx2track.items()}
print(len(idx2track))


# In[77]:


vector = track_feature_copy.iloc[:,1:].values


# In[78]:


vector.shape


# In[79]:


# save the vector and map so the neural network can use it.
import pickle
with open (output_dir + "music_vector_all.pkl", "wb") as f:
    pickle.dump(vector, f)
with open (output_dir + "track2idx_all.pkl", "wb") as f:
    pickle.dump(track2idx, f)

