import numpy as np
import os
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Activation, Flatten
from keras.layers.convolutional import Convolution1D as Conv1D
from keras.layers.pooling import MaxPooling1D as Max1D
from keras.layers.pooling import GlobalMaxPooling1D as GlobalMax1D
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense
from multiprocessing import Process as Thread, Queue
from itertools import cycle, islice

EPOCH_SIZE = 16384
train_queue = Queue(maxsize=20)
test_queue = Queue(maxsize=20)


def generate_random_string(length):
    return os.urandom(length)


def principal_period(s):
    i = (s+s).find(s, 1, -1)
    return None if i == -1 else s[:i]


def make_dataset(dataset, nsamp, slen, maxkeylen):
    x = np.zeros((nsamp, slen, 8))
    y = np.zeros((nsamp, maxkeylen))

    for i in xrange(nsamp):
        keylen = np.random.randint(maxkeylen) + 1

        # save key len as categorical variable
        y[i, keylen - 1] = 1.0

        dataptr = np.random.randint(len(dataset) - slen)
        data = dataset[dataptr:dataptr + slen]
        data = np.fromstring(data, dtype=np.uint8)

        key = generate_random_string(keylen)
        while principal_period(key) is not None:
            key = generate_random_string(keylen)

        key = np.fromstring(key, dtype=np.uint8)

        key_nrep = int(np.ceil(float(slen) / float(len(key))))
        key_exp = np.tile(key, key_nrep)[:slen]

        xor_ciphertext = np.bitwise_xor(data, key_exp)

        x[i, :, :] = np.unpackbits(xor_ciphertext).reshape(slen, 8)

    return x, y


def generate_data(training, data_set, slen, maxkeylen):
    while True:
        ds = make_dataset(data_set, EPOCH_SIZE, slen, maxkeylen)
        if training:
            train_queue.put(ds)
        else:
            test_queue.put(ds)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        prog=__file__,
        description="Trains deep learning model to predict the length of an XOR key",
    )

    parser.add_argument('-c', '--conv', default=False, required=False, action='store_true',
                        help='Use convolutional neural network layers instead of LSTM layers')
    parser.add_argument('--output-dim', type=int, default=256, required=False, help='Output dimensions of LSTM/CNN')
    parser.add_argument(
        '--activation',
        default="relu",
        required=False,
        help='Activation function for convolutional layers',
        choices=[
            "softmax",
            "relu",
            "softplus",
            "softsign",
            "tanh",
            "sigmoid",
            "hard_sigmoid",
            "linear"
        ]
    )
    parser.add_argument(
        '--output-activation',
        default="softmax",
        required=False,
        help='Activation function for output layer',
        choices=[
            "softmax",
            "relu",
            "softplus",
            "softsign",
            "tanh",
            "sigmoid",
            "hard_sigmoid",
            "linear"
        ]
    )

    args = parser.parse_args()
    print args.__dict__

    lr = 0.001
    maxkeylen = 32
    slen = maxkeylen * 2

    print "Loading data"
    with open("enwik8", "r") as f:
        dataset = f.read()

    tr_dataset = dataset[:9000000]
    val_dataset = dataset[9000000:]

    print "Starting vectorization threads"
    for _ in xrange(4):
        train_thread = Thread(target=generate_data, args=(True, tr_dataset, slen, maxkeylen))
        train_thread.daemon = True
        train_thread.start()

    for _ in xrange(4):
        test_thread = Thread(target=generate_data, args=(False, val_dataset, slen, maxkeylen))
        test_thread.daemon = True
        test_thread.start()

    do_conv = args.conv
    do_lstm = not do_conv

    input_layer = Input(shape=(slen, 8))
    prev_layer = input_layer

    if do_lstm:
        lstm1_layer = LSTM(args.output_dim, return_sequences=True)(prev_layer)
        lstm2_layer = LSTM(args.output_dim, return_sequences=True, go_backwards=True)(lstm1_layer)
        gmp_layer = GlobalMax1D()(lstm2_layer)
    else:
        for _ in xrange(4):
            conv_layer = Conv1D(args.output_dim, 16, border_mode='same')(prev_layer)
            bn_layer = BatchNormalization(axis=2)(conv_layer)
            act_layer = Activation(args.activation)(bn_layer)
            mp_layer = Max1D(pool_length=2)(act_layer)
            prev_layer = mp_layer

        gmp_layer = GlobalMax1D()(prev_layer)

    output_layer = Dense(maxkeylen, activation=args.output_activation)(gmp_layer)

    model = Model(input=input_layer, output=output_layer)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(lr=lr))

    model_name = "{0}-lr-{1}-od-{2}-oa-{3}-a-{4}.model".format(
        "lstm" if do_lstm else "cnn",
        lr,
        args.output_dim,
        args.output_activation,
        args.activation
    )

    print "Model Summary"
    model.summary()
    model.save(model_name)

    x, y = test_queue.get()

    y_pred = model.predict(x)
    best_validation_score = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))

    nepoch = 1
    stale_epochs = 0
    while True:
        x, y = train_queue.get()
        model.fit(x, y, nb_epoch=1, verbose=1)
        nepoch += 1

        x, y = test_queue.get()
        y_pred = model.predict(x)

        validation_score = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
        if validation_score > best_validation_score:
            best_validation_score = validation_score
            print "New best validation score: {0} (saving)".format(validation_score)
            stale_epochs = 0
            model.save(model_name)
        else:
            print "Validation score: {0}".format(validation_score)
            stale_epochs += 1

        if stale_epochs > 10:
            stale_epochs = 0
            lr = float(0.9 * model.optimizer.lr.get_value())
            model.optimizer.lr.set_value(lr)
            print "Reducing learning rate to {0} after 10 stale epochs".format(lr)
