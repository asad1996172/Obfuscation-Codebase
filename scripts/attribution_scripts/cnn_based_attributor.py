"""
Module for training CNN based attribution classifier

Usage
-------
python3 cnn_based_attributor.py \
    --dataset-path ../../data_files/datasets/prepared_datasets/ebg/experiment_0_5 \
    --trained-model-path ../../data_files/attribution_models/ebg/experiment_0_5/cnn
"""
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils.vis_utils import plot_model
from keras.layers.merge import concatenate
from keras.callbacks import TensorBoard, EarlyStopping
import os
import numpy as np
import sys
import argparse
import os
import operator
import io
from sklearn.model_selection import train_test_split
import collections
from prettytable import PrettyTable
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn import preprocessing
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def fill_in_missing_words_with_zeros(embeddings_index, word_index, EMBEDDING_DIM):
    """
    Function for filling in for the rare words embeddings with 0s
    """
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def load_model(filename):
    embeddings_index = {}
    f = open(filename)
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
            if len(coefs) == 300:
                embeddings_index[word] = coefs
        except:
            # print(values)
            c = 1
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

def create_tokenizer(lines):
    tokenizer = Tokenizer(
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True)
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_length(lines):
    return max([len(s.split()) for s in lines])

def save_dict(filename, dictionary):
    with open(filename + '.pickle', 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_dict(filename):
    with open(filename + '.pickle', 'rb') as handle:
        b = pickle.load(handle)
    return b

def setting_up_data(data):
    x_train, x_test, y_train, y_test = data
    authors_limit = len(list(set(y_train)))

    # calculate max document length
    MAX_SEQUENCE_LENGTH = max_length(x_train)

    # create tokenizer
    tokenizer = create_tokenizer(x_train)
    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)

    # calculate vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    print('\nMax document length: %d' % MAX_SEQUENCE_LENGTH)
    print('Vocabulary size: %d' % vocab_size)

    # padding sequences to make them of the same length
    x_train = pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH)
    x_test = pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH)

    # converting target labels to categorical
    y_train = to_categorical(y_train, num_classes=authors_limit)
    y_test = to_categorical(y_test, num_classes=authors_limit)
    print('\nShape of training data tensor:', x_train.shape)
    print('Shape of training label tensor:', y_train.shape)

    # loading word embeddings
    EMBEDDING_DIM = 300
    word_index = tokenizer.word_index
    print("\nloading GLOVE model ...")
    embedding_matrix = load_model('../../data_files/glove.840B.300d.txt')
    # embedding_matrix = {}
    print("Filling non existing words ...")
    embedding_matrix = fill_in_missing_words_with_zeros(
        embedding_matrix, tokenizer.word_index, EMBEDDING_DIM)

    return MAX_SEQUENCE_LENGTH, tokenizer, vocab_size, EMBEDDING_DIM, embedding_matrix, np.array(x_train), \
        np.array(x_test), np.array(
            y_train), np.array(y_test), authors_limit

def get_data(dataset_path):

    with open('{}/X_train.pickle'.format(dataset_path), 'rb') as handle:
        all_train = pickle.load(handle)

    with open('{}/X_test.pickle'.format(dataset_path), 'rb') as handle:
        all_test = pickle.load(handle)

    x_train = []
    y_train = []

    x_test = []
    y_test = []

    # data format in pickled file => (file_path, filename, author_id, author, input_text)

    for (_, _, author_id, _, input_text) in all_train:
        x_train.append(input_text)
        y_train.append(author_id)

    for (_, _, author_id, _, input_text) in all_test:        
        x_test.append(input_text)
        y_test.append(author_id)

    return x_train, x_test, y_train, y_test


def save_models(required_information, model, trained_model_path):
    save_model_directory = trained_model_path
    if not os.path.exists(save_model_directory):
        os.makedirs(save_model_directory)

    save_dict(save_model_directory +
                    '/required_meta', required_information)
    model.save(save_model_directory + '/model.h5')


def define_model_CNN_word_word(authors_limit, embedding_matrix, vocab_size, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH):
    inputs1 = Input(shape=(MAX_SEQUENCE_LENGTH,))
    embed1 = Embedding(vocab_size,
                       EMBEDDING_DIM,
                       weights=[embedding_matrix],
                       input_length=MAX_SEQUENCE_LENGTH,
                       trainable=False)(inputs1)
    inputs2 = Input(shape=(MAX_SEQUENCE_LENGTH,))
    embed2 = Embedding(vocab_size,
                       EMBEDDING_DIM,
                       weights=[embedding_matrix],
                       input_length=MAX_SEQUENCE_LENGTH,
                       trainable=True)(inputs2)

    merged = concatenate([embed1, embed2])

    conv1 = Conv1D(64, 5, activation="relu")(merged)
    poo1 = MaxPooling1D(MAX_SEQUENCE_LENGTH - 5 + 1)(conv1)
    flatten1 = Flatten()(poo1)
    dense1 = Dense(256, activation="relu")(flatten1)
    dropout1 = Dropout(0.5)(dense1)
    outputs = Dense(int(authors_limit), activation="softmax")(dropout1)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
    # summarize
    print(model.summary())

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset-path', required=True, type=str)
    parser.add_argument('-tmp', '--trained-model-path',
                        required=True, type=str)
    args = parser.parse_args()

    data = get_data(args.dataset_path)

    MAX_SEQUENCE_LENGTH, tokenizer, vocab_size, EMBEDDING_DIM, embedding_matrix, X_train, X_test, \
        y_train, y_test, authors_limit = setting_up_data(data)

    print("\nStarting Training ...")
    model = define_model_CNN_word_word(
        authors_limit, embedding_matrix, vocab_size, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH)

    earlyStopping = EarlyStopping(
        monitor='train_loss', patience=10, verbose=0, mode='min')
    training_history = model.fit(
        [X_train, X_train],
        y_train,
        epochs=15,
        batch_size=50,
        callbacks=[earlyStopping]
    )

    required_information = {
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "tokenizer": tokenizer,
        "vocab_size": vocab_size,
        "embedding_dim": EMBEDDING_DIM,
        "authors_limit": authors_limit
    }

    save_models(required_information, model, args.trained_model_path)

    loss, acc = model.evaluate([X_test, X_test], y_test, verbose=0)
    print('Test Accuracy: %f' % (acc * 100))
