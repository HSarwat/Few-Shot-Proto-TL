import sys
import os
from matplotlib.pyplot import axis
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from processing.constants import *
from os.path import join as pjoin
import numpy as np
import pickle
from sklearn.model_selection import KFold
import pathlib
pathlib.Path(PICKLES_PATH).mkdir(exist_ok=True)
import logging
from joblib import load
from sklearn.feature_selection import SelectKBest, f_classif
from tqdm import tqdm
import itertools



'''This file reads the input pickle file'''
# If you have your own data, you can replace this function with your own
class data_structure():
    def __init__(self, pickle_name="data.pkl") -> None:
        self.data_dict = {}
        self.pickle_name = pjoin(PICKLES_PATH, pickle_name)
        self.tensor_list = ["FMG", "IMU", "EMG"]
        if os.path.isfile(self.pickle_name):
            self.data_dict = pickle.load(open(self.pickle_name, "rb"))
        else:
            print("No data structure pickle")


    def save_to_pickle(self):
        pickle.dump(self.data_dict, open(self.pickle_name, "wb"))
        logging.info("data_dict saved to a pickle")

    def generate_test_train(self, train_subs, test_subs, actions=ALL_ACTION_LIST, ):
        if self.data_dict["LABEL_NAME"] is None:
            raise IndexError(
                "There is no label type in the data dictionary, so you will need to add one by using one of the \'update\' methods")

        X_train, y_train = self.dictionary_to_list(train_subs, actions)
        X_test, y_test = self.dictionary_to_list(test_subs, actions)

        return X_train, X_test, y_train, y_test

    def dictionary_to_list(self, subs, actions):
        X = []
        y = []
        if len(subs) == 0:
            return X, y
        for sub in subs:
            for action in actions:
                for trial in ALL_TRIAL_LIST:
                    data_len = self.data_dict[sub][action][trial]["IMU"].shape[0]
                    X.append([self.data_dict[sub][action][trial][key] for key in self.tensor_list])
                    y.append(np.tile(self.data_dict[sub][action][trial]["LABEL"], reps=(data_len, 1)))
        FMG_input = np.vstack([X[i][0] for i in range(len(X))]).astype("float")
        IMU_input = np.vstack([X[i][1] for i in range(len(X))]).astype("float")
        EMG_input = np.vstack([X[i][2] for i in range(len(X))]).astype("float")

        X = {"EMG_input": EMG_input,
             "FMG_input": FMG_input,
             "IMU_input": IMU_input}
        y = np.vstack(y)
        assert FMG_input.shape[0] == IMU_input.shape[0] == EMG_input.shape[0] == y.shape[0], "Number of rows is off"
        return X, y

    def cross_validation_generator(self, subs, folds, shuffle=False, actions=ALL_ACTION_LIST):
        kf = KFold(folds, shuffle=shuffle)
        fold_num = 0
        for train_subs_indices, test_subs_indices in kf.split(subs):
            fold_num += 1
            train_subs = [subs[i] for i in train_subs_indices]
            test_subs = [subs[i] for i in test_subs_indices]
            X_train, X_test, y_train, y_test = self.generate_test_train(train_subs, test_subs, actions)
            yield X_train, X_test, y_train, y_test, test_subs, train_subs, fold_num


    # This is the dimensionality reduction function
    def dimensionality_reduction(self, fn):
        X_train, y_train = self.dictionary_to_list([9], ALL_ACTION_LIST)

        train_y = np.ravel(y_train)
        train_X = sensor_selection(X_train, "IFE")

        scaler = load(f'{root_dir}\Data\scaler.bin')
        train_X = scaler.transform(train_X)

        features = SelectKBest(f_classif, k=fn).fit(train_X, train_y)
        return features

def sensor_selection(input_data, sensors):
    data = []
    if "I" in sensors:
        data.append(input_data["IMU_input"])
    if "F" in sensors:
        data.append(input_data["FMG_input"])
    if "E" in sensors:
        data.append(input_data["EMG_input"])
    X = np.hstack(data)
    return X


