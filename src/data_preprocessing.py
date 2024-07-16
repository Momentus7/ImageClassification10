import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def preprocess_data(X_train, X_test, y_train, y_test):
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    mean = np.mean(X_train)
    std = np.std(X_train)

    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return X_train, X_test, y_train, y_test

def split_data(X_train, y_train):
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=0)
    return X_train, X_valid, y_train, y_valid
