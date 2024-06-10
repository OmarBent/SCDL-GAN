"""
Created on Wed Mar 20 13:12:32 2019

@author: bentanfous
"""
import numpy as np
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
from keras.layers import Dropout
from keras.layers import Bidirectional
import scipy.io as sio


def bi_lstm(opt, data, subject_labels, action_labels):


    # Split data into training and testing
    nb_sequences = len(data)
    feature_size = np.asarray(data[0]).shape[0]
    lengths = []
    for x in data:
        lengths.append(len(x))
    maxlen = np.max(lengths)
    nb_classes = opt.n_classes

    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for j in range(0, nb_sequences):
        if subject_labels[j] in opt.train_subjects:
            x_train.append(np.transpose(data[j]))
            y_train.append(np.transpose(action_labels[j])-1)
        else:
            x_test.append(np.transpose(data[j]))
            y_test.append(np.transpose(action_labels[j])-1)

    x_train = sequence.pad_sequences(x_train, maxlen=maxlen, padding='post', dtype='float32')
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen, padding='post', dtype='float32')

    dummy_y = np_utils.to_categorical(y_train, nb_classes)
    dummy_y_test = np_utils.to_categorical(y_test, nb_classes)

    # Model
    model = Sequential()
    model.add(Bidirectional(LSTM(opt.lstm_size, return_sequences=False), input_shape=(maxlen, feature_size)))
    model.add(Dropout(opt.dropout_prob))
    model.add(Dense(nb_classes, input_dim=opt.lstm_size, activation='softmax', name='dense'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    # train model TODO set validation data from training data
    model.fit(x_train, dummy_y,  validation_data=(x_test, dummy_y_test), epochs=opt.nb_epochs, batch_size=opt.b_size, shuffle=True)
    # test model
    scores = model.evaluate(x_test, dummy_y_test, batch_size=opt.b_size, verbose=0)
    accuracy = model.predict_classes(x_test)

    return accuracy, scores


