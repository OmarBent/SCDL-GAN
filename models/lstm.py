import numpy as np
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from keras.utils import to_categorical


def bi_lstm(opt, data, subject_labels, action_labels):
    """
    Bidirectional LSTM classifier for sequence classification.

    Args:
        opt (Namespace): Options with attributes:
            - n_classes (int): Number of action classes.
            - lstm_size (int): Hidden size of LSTM layer.
            - dropout_prob (float): Dropout rate.
            - nb_epochs (int): Number of training epochs.
            - b_size (int): Batch size.
            - train_subjects (list): IDs of training subjects.
        data (list of np.ndarray): List of shape sequences (each shape: [features x time]).
        subject_labels (list): Subject ID for each sequence.
        action_labels (list): Ground truth label for each sequence.

    Returns:
        Tuple: (predicted_labels, [test_loss, test_accuracy])
    """
    nb_sequences = len(data)
    feature_size = data[0].shape[0]
    maxlen = max(len(seq[0]) for seq in data)
    nb_classes = opt.n_classes

    x_train, y_train, x_test, y_test = [], [], [], []

    for i in range(nb_sequences):
        sequence_data = np.transpose(data[i])  # Now shape is (T, F)
        label = action_labels[i] - 1  # Class indices should start from 0

        if subject_labels[i] in opt.train_subjects:
            x_train.append(sequence_data)
            y_train.append(label)
        else:
            x_test.append(sequence_data)
            y_test.append(label)

    x_train = sequence.pad_sequences(x_train, maxlen=maxlen, padding='post', dtype='float32')
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen, padding='post', dtype='float32')

    y_train_cat = to_categorical(y_train, num_classes=nb_classes)
    y_test_cat = to_categorical(y_test, num_classes=nb_classes)

    # Model definition
    model = Sequential()
    model.add(Bidirectional(LSTM(opt.lstm_size, return_sequences=False), input_shape=(maxlen, feature_size)))
    model.add(Dropout(opt.dropout_prob))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    # Training
    model.fit(
        x_train, y_train_cat,
        validation_data=(x_test, y_test_cat),
        epochs=opt.nb_epochs,
        batch_size=opt.b_size,
        shuffle=True
    )

    # Evaluation
    scores = model.evaluate(x_test, y_test_cat, batch_size=opt.b_size, verbose=0)
    predictions = np.argmax(model.predict(x_test), axis=1)

    return predictions, scores
