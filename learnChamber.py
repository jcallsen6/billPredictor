'''
Trains a neural network to learn a chamber of congress
'''

import numpy as np
import keras
from keras.models import load_model, save_model
from keras import layers
import pandas as pd
import argparse
import pickle
import os


def get_args():
    '''
    Gets args for this program

    param:
    None

    return:
    parsed arguments object
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('data', default='data/',
                        help='Relative path for csv of data to train on', type=str)
    parser.add_argument('name', help='Name of model', type=str)

    parser.add_argument('-epochs', default=10,
                        help='Number epochs to train for', type=int)
    parser.add_argument('-batch_size', default=32,
                        help='Batch size for training neural network', type=int)
    parser.add_argument(
        '-name', help='Relative path for model .h5 file to start from', type=str)
    parser.add_argument('-new_model', action='store_true',
                        help='Start over and train a new model')
    parser.add_argument('-learning_rate', default=0.01,
                        help='Initial learning rate for training', type=float)
    parser.add_argument(
        '-max_len', default=2000, help='Max length of subjects list in characters', type=int)
    parser.add_argument(
        '-embedding_path', help='Relative path for pretrained texet embedding weights to start from', type=str)
    parser.add_argument('-embedding_dim', default=50,
                        help='Output number of dimensions for text embedding', type=int)

    return(parser.parse_args())


def encode_data(csv_path, max_len, model_name):
    '''
    Reads in dataframe and encodes all the non-numerical features and labels
    source: https://realpython.com/python-keras-text-classification/#word-embeddings

    param:
    csv - string path to dataframe saved as a csv
    max_len - int max length of a text sequence
    model_name - str path to model

    return:
    data - numpy 2d array of transformed characteristic data
    encoded_text - numpy 2d array of encoded text data
    labels - numpy 1d array of binary labels
    word_index - word index of tokenizer
    max_len - length of longest sequence of subject
    party_indicies - tuple of party classes for (D, R, I)
    '''
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import LabelBinarizer
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences

    df = pd.read_csv(csv_path)

    # encode labels
    outcome_list = df['Outcome'].tolist()
    labels = np.array([1 if ('PASSED' in outcome) or (
        'ENACTED' in outcome) else 0 for outcome in outcome_list])
    df.drop(['Outcome'], axis=1, inplace=True)

    # encode features
    party_encoder = LabelEncoder()
    party_encoder.fit(df['Sponsor Party'])
    df['Sponsor Party'] = party_encoder.transform(df['Sponsor Party'])
    party_indicies = (
        party_encoder.transform(['D'])[0], party_encoder.transform(['R'])[0], party_encoder.transform(['I'])[0])

    subjects_list = df['Subjects'].astype(str).values
    tokenizer = Tokenizer(len(np.unique(subjects_list)))
    tokenizer.fit_on_texts(subjects_list)

    # save off tokenizer for future predictions
    with open(model_name + '/subjects.tokenizer', 'wb') as token_file:
        pickle.dump(tokenizer, token_file)

    word_index = tokenizer.word_index

    encoded_text = tokenizer.texts_to_sequences(subjects_list)
    encoded_text = pad_sequences(
        encoded_text, padding='post', maxlen=max_len)

    df.drop(['Subjects'], axis=1, inplace=True)

    return df.to_numpy(), encoded_text, labels, word_index, max_len, party_indicies


def split_data(char_features, text_features, labels):
    '''
    Splits data into training and testing sets

    param:
    char_features - 2d numpy array of characteristic features
    text_features - 2d numpy array of encoded text features
    labels - 2d numpy array of labels

    return:
    train_char_features - numpy array of characteristic training features
    train_text_features - numpy array of text training features
    train_labels - numpy array of training labels
    test_char_features - numpy array of training features
    test_text_features - numpy array of text training features
    test_labels - numpy array of training labels
    '''
    # shuffle features and labels randomly
    indx = np.random.permutation(len(labels))

    char_features = char_features[indx]
    text_features = text_features[indx]
    labels = labels[indx]

    indx = int(0.8 * len(labels))

    train_char_features = char_features[:indx]
    test_char_features = char_features[indx:]

    train_text_features = text_features[:indx]
    test_text_features = text_features[indx:]

    train_labels = labels[:indx]
    test_labels = labels[indx:]

    return train_char_features, train_text_features, train_labels, test_char_features, test_text_features, test_labels


def create_embedding(pretrained_path, word_index, embedding_dim):
    '''
    Loads and creates embedding matrix from pretrained embeddings
    source: https://realpython.com/python-keras-text-classification/#word-embeddings

    param:
    pretrained_path - string path to pretrained embedding matrix
    word_index - int word index from tokenizer
    embedding_dim - int embedding dimension used for model

    return:
    embedding_matrix - numpy 2d array of embedding matrix weights
    '''
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(pretrained_path) as data:
        for line in data:
            word, *vector = line.split()
            if word in word_index:
                indx = word_index[word]
                embedding_matrix[indx] = np.array(vector, dtype=np.float32)[
                    : embedding_dim]

    return embedding_matrix


def build_model(input_shape, lr, embedding_matrix=None):
    '''
    Creates a keras model to learn the dataset

    param:
    input_shape - tuple shape of input data, (characteristics size, text size, vocab size,
                                              max length of sequences, embedding dim)
    lr - decimal for initial learning rate
    embedding_matrix - numpy 2d array of pretrained text embeddings

    return:
    model - keras model
    '''
    from keras.layers import Input, Dense, Concatenate, Embedding, GlobalMaxPooling1D, Conv1D, Bidirectional, GRU, Dropout
    from keras.models import Model

    char_input = Input((input_shape[0],))
    text_input = Input((input_shape[1],))

    # if using embedding matrix convert to a list for keras api
    try:
        if(embedding_matrix.all() != None):
            embedding_matrix = [embedding_matrix]
    except:
        pass

    embedding = Embedding(input_length=input_shape[3], input_dim=input_shape[2],
                          weights=embedding_matrix, output_dim=input_shape[4], trainable=True)(text_input)
    conv_layer = Conv1D(32, (3,), activation='relu')(embedding)
    maxpool_text = GlobalMaxPooling1D()(conv_layer)

    dense_char = Dense(input_shape[0], activation='relu')(char_input)
    dense_text = Dense(input_shape[1]//4, activation='relu')(maxpool_text)

    concat = Concatenate()([dense_char, dense_text])
    dropout_layer = Dropout(0.1)(concat)
    dense_combined = Dense(15, activation='relu')(dropout_layer)

    # binary classification
    output_layer = Dense(1, activation='sigmoid')(dense_combined)

    model = Model(inputs=[char_input, text_input], outputs=output_layer)

    opt = keras.optimizers.Adamax(learning_rate=lr)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt, metrics=['accuracy'])

    return model


def plot_history(history):
    '''
    Plot training history

    param:
    history - history dict containing loss per metric for each epoch

    return:
    None
    '''
    import matplotlib.pyplot as plt

    epochs = np.linspace(0, args.epochs, args.epochs)

    plt.figure('Training History')
    epochs = np.linspace(0, len(history['loss']), len(history['loss']))

    plt.subplot(2, 1, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, history['loss'], label='Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, history['accuracy'], label='Accuracy')
    plt.plot(epochs, history['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.savefig('history.png')
    plt.show()


def eval_model(model, test_char_features, test_text_features, test_labels, party_indicies):
    '''
    Evlaulates a given model on the given test set

    param:
    model - keras model to evlaulate
    test_char_features - char features numpy array
    test_text_features - encoded texte features numpy array
    test_labels - numpy array of class labels
    party_indicies - tuple of indicies for each party(D, R, I)

    return:
    best float threshold to use
    '''
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, log_loss
    import matplotlib.pyplot as plt

    result = model.evaluate(
        [test_char_features, test_text_features], test_labels)
    print('Testing Results:')
    print(dict(zip(model.metrics_names, result)))

    test_pred = model.predict([test_char_features, test_text_features])
    fpr, tpr, thresholds = roc_curve(test_labels, test_pred)

    roc_auc = auc(fpr, tpr)
    with np.errstate(divide='ignore', invalid='ignore'):
        performance = np.true_divide(tpr, fpr)
        performance[np.isinf(performance)] = 0
        max_threshold = thresholds[np.nanargmax(performance)]

    plt.figure('ROC Curve')
    plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.plot([0, 1], [0, 1], label='Random Guessing')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend()
    plt.show()

    precision, recall, thresholds = precision_recall_curve(
        test_labels, test_pred)
    average_precision = average_precision_score(test_labels, test_pred)
    random_precision = np.average(test_labels)
    plt.figure('Precision vs Recall')
    plt.plot(recall, precision, label='AP = %0.2f' % average_precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.hlines(random_precision, 0, 1, label='Random Guessing')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend()

    plt.show()

    # Error vs party introduced
    test_pred = test_pred.flatten()
    # Done to avoid log error division by zero
    test_pred[test_pred == 1] = 0.99999
    binary_crossentropy = log_loss(test_labels, test_pred)
    error_df = pd.DataFrame(
        {'Sponsor Party': test_char_features[:, 3], 'Error': binary_crossentropy})

    error_df_d = error_df[error_df['Sponsor Party'] == party_indicies[0]]
    error_df_r = error_df[error_df['Sponsor Party'] == party_indicies[1]]
    error_df_i = error_df[error_df['Sponsor Party'] == party_indicies[2]]

    errors = [error_df_d['Error'].mean(), error_df_r['Error'].mean(),
              error_df_i['Error'].mean()]
    parties = ['Democrat', 'Republican', 'Independant']

    plt.figure('Error vs Sponsor Party')
    plt.bar(parties, errors)
    plt.ylabel('Average Binary Cross Entropy Error')
    plt.show()

    return max_threshold


if __name__ == '__main__':
    args = get_args()

    if(args.embedding_path != None):
        args.embedding_dim = int(args.embedding_path.split('.')[2][:-1])

    if(not os.path.exists(args.name)):
        os.mkdir(args.name)

    char_features, text_features, labels, word_index, max_len, party_indicies = encode_data(
        args.data, args.max_len, args.name)

    vocab_size = len(word_index) + 1

    embedding_matrix = None
    if(args.embedding_path != None):
        embedding_matrix = create_embedding(
            args.embedding_path, word_index, args.embedding_dim)

    train_char_features, train_text_features, train_labels, test_char_features, test_text_features, test_labels = split_data(
        char_features, text_features, labels)

    if(args.new_model):
        if(not os.path.exists(args.name)):
            os.mkdir(args.name)

        data_shape = (
            train_char_features.shape[1], train_text_features.shape[1], vocab_size, max_len, args.embedding_dim)
        model = build_model(data_shape, args.learning_rate, embedding_matrix)
    else:
        model = load_model(args.name + '/model.h5')
    print(model.summary())

    my_callbacks = [
        keras.callbacks.EarlyStopping(patience=10),
        keras.callbacks.ModelCheckpoint(
            filepath=args.name + '/model.h5'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                          patience=3, min_lr=args.learning_rate/100)
    ]

    imbalance = np.average(train_labels)
    class_weights = {0: 1/imbalance, 1: 1}

    history = model.fit([train_char_features, train_text_features], train_labels, batch_size=args.batch_size,
                        epochs=args.epochs, verbose=1, validation_split=0.2, callbacks=my_callbacks, class_weight=class_weights)

    if(args.epochs > 1):
        plot_history(history.history)

    thresh = eval_model(model, test_char_features,
                        test_text_features, test_labels, party_indicies)

    with open(args.name + '/params', 'wb') as thresh_file:
        model_params = {'thresh': thresh, 'max_len': max_len,
                        'party_indicies': party_indicies}
        pickle.dump(model_params, thresh_file)
