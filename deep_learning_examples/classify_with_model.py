import numpy as np
import os
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers import Input, Activation, Flatten
from keras.layers.convolutional import Convolution1D as Conv1D
from keras.layers.pooling import MaxPooling1D as Max1D
from keras.layers.pooling import GlobalMaxPooling1D as GlobalMax1D
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense
from multiprocessing import Process as Thread, Queue
from itertools import cycle, islice



if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        prog=__file__,
        description="Classify the XOR key size with deep learning model",
    )

    parser.add_argument('-m', '--model', required=True, help="File path to model to use")
    parser.add_argument('-e', '--encrypted', required=True, help="File path to encrypted data")
    args = parser.parse_args()

    print "Loading model"
    model = load_model(args.model)
    model.summary()

    with open(args.encrypted, "rb") as f:
        data = f.read()

    print "Encrypted data:", repr(data)
    data = np.fromstring(data, dtype=np.uint8)
    data = np.unpackbits(data).reshape(1, data.shape[0], 8)

    prediction = model.predict(data)
    print "Predicts {0} length".format(prediction.argmax() + 1)
    print prediction
