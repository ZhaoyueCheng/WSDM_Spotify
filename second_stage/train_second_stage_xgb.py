# compute original accuracy
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
import xgboost

valid_target = pickle.load(open("blend_valid_target", "rb"))
valid = pickle.load(open("blend_valid", "rb"))

scores = (np.sum(valid[:, :-1], axis = 1) > 3).astype(int)
print(accuracy_score(valid_target, scores))


train_target = np.load("blend_train_target.npy")
train = np.load("blend_train.npy")

scores = (np.sum(train[:, :-1], axis = 1) > 3).astype(int)
print(accuracy_score(train_target, scores))

from catboost import CatBoostClassifier, Pool
p = Pool(train, label=train_target, cat_features=[6])

cb_model = CatBoostClassifier(iterations=100,
                              eval_metric='Accuracy',
                              learning_rate=0.02,
                              max_depth=5)



pred = model.predict(valid)


print(accuracy_score(pred, valid_target))