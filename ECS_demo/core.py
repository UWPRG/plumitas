import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.models import load_model
from os.path import dirname, join
import sys
import time
import statistics


def load_data(data_file_name, h5File=False):
    """
    Loads data from module_path/data/data_file_name.

    Parameters
    ----------
    data_file_name : string
        name of csv file to be loaded from module_path/data/
        data_file_name.
    h5File : boolean, optional, default = False
        if True opens hdf5 file

    Returns
    -------
    data : Pandas DataFrame
    """
    module_path = dirname(__file__)
    if h5File:
        data = load_model(join(module_path, 'data', data_file_name))
    else:
        with open(join(module_path, 'data', data_file_name), 'rb') as csv_file:
            data = pd.read_csv(csv_file, encoding='latin1')
    return data


def data_setup(top_words=1000, max_words=150):
    """
    preprocesses the twitter climate data. Does things like changes output
    to one hot encoding, performs word embedding/padding
    Parameters
    ----------
    top_words: int
      number of words to include in model
    max_words:
      max number of words to have in tweet

    Returns
    -------
    X, Y : array
      X and Y arrays of data
    """
    data = load_data("tweet_global_warming.csv")
    print("Full dataset: {}".format(data.shape[0]))
    data['existence'].fillna(value='ambiguous',
                             inplace=True)  # replace NA's in existence with "ambiguous"
    data['existence'].replace(('Y', 'N'), ('Yes', 'No'),
                              inplace=True)  # rename so encoder doesnt get confused
    data = data.dropna()  # now drop NA values
    print("dataset without NaN: {}".format(data.shape[0]))
    X = data.iloc[:, 0]
    Y = data.iloc[:, 1]
    print("Number of unique words: {}".format(len(np.unique(np.hstack(X)))))

    # one hot encoding = dummy vars from categorical var
    # Create a one-hot encoded binary matrix
    # N, Y, Ambig
    # 1, 0, 0
    # 0, 1, 0
    # 0, 0, 1

    # encode class as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)

    # convert integers to one hot encoded
    Y = np_utils.to_categorical(encoded_Y)

    # convert X to ints (y is already done)
    token = Tokenizer(num_words=top_words,
                      filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True,
                      split=' ', char_level=False, oov_token=None)
    token.fit_on_texts(texts=X)
    X = token.texts_to_sequences(texts=X)
    X = sequence.pad_sequences(X, maxlen=max_words)
    return X, Y

def baseline_model(top_words=1000, max_words=150, filters=32):
    """
    a really bad docstring!!! But look how it self-formats!
    Parameters
    ----------
    data_file_name : string
        name of csv file to be loaded from module_path/data/
        data_file_name.
    h5File : boolean, optional, default = False
        if True opens hdf5 file

    Returns
    -------
    model : keras model object
    """
    model = Sequential()
    model.add(Embedding(top_words + 1, filters,
                        input_length=max_words))  # is it better to preconvert using word to vec?
    model.add(Convolution1D(filters=filters, kernel_size=3, padding='same',
                            activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model

class Benchmark:
    """
    benchmark method used by the unittests
    """
    @staticmethod
    def run(function):
        timings = []
        stdout = sys.stdout
        for i in range(5):
            sys.stdout = None
            startTime = time.time()
            function()
            seconds = time.time() - startTime
            sys.stdout = stdout
            timings.append(seconds)
            mean = statistics.mean(timings)
            print("{} {:3.2f} {:3.2f}".format(
                1 + i, mean,
                statistics.stdev(timings, mean) if i > 1 else 0))
