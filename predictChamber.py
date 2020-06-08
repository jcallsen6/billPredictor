'''
Predicts using a neural network for a chamber of congress
'''

import numpy as np
import pickle
import keras
from keras.models import load_model, save_model
import pandas as pd
import argparse


def get_args():
    '''
    Gets args for this program

    param:
    None

    return:
    parsed arguments object
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('input_data', default='data/',
                        help='Path for csv of data to train on', type=str)
    parser.add_argument('output_data', default='data/',
                        help='Path for csv of data to train on', type=str)
    parser.add_argument('path', help='Path to of model folder', type=str)
    parser.add_argument('-api_key', default='apir_key.json',
                        help='Path to json file containing api key')
    parser.add_argument('-member_char', default='data/congress_mem_char.json',
                        help='Path to csv of member characteristics')

    return(parser.parse_args())


def encode_data_pred(df, max_len, party_indicies, model_path):
    '''
    Encodes unlabeled df for prediction

    param:
    df - pandas dataframe containing features to encode
    max_len - max length of a text sequence
    party_indicies - int tuple for numerical values of (D, R, I)
    model_path - str path to model folder for encoders

    return:
    char_features - np array bill characteristics
    text_features - np array of encoded bill subjects
    '''
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences

    sponsor_parties = df['Sponsor Party'].to_numpy()
    parties_dict = {
        'D': party_indicies[0], 'R': party_indicies[1], 'I': party_indicies[2]}
    sponsor_parties = np.vectorize(parties_dict.get)(sponsor_parties)
    df['Sponsor Party'] = sponsor_parties

    subjects_list = df['Subjects'].astype(str).values
    with open(model_path + '/subjects.tokenizer', 'rb') as token_file:
        tokenizer = pickle.load(token_file)

        encoded_text = tokenizer.texts_to_sequences(subjects_list)
        encoded_text = pad_sequences(
            encoded_text, padding='post', maxlen=max_len)

        df.drop(['Subjects'], axis=1, inplace=True)
        df.drop('Outcome', axis=1, inplace=True)

    return df.to_numpy(), encoded_text


if __name__ == '__main__':
    args = get_args()

    with open(args.path + '/params', 'rb') as params_file:
        params_dict = pickle.load(params_file)
        threshold = params_dict['thresh']
        party_indicies = params_dict['party_indicies']
        max_len = params_dict['max_len']

    pred_data = pd.read_csv(args.input_data)

    char, text = encode_data_pred(
        pred_data, max_len, party_indicies, args.path)

    model = load_model(args.path + '/model.h5')
    predictions = model.predict([char, text])
    predictions[predictions < threshold] = 0
    predictions[predictions >= threshold] = 1

    pass_dict = {0: 'Fail', 1: 'Pass'}
    slugs = pred_data['bill_slug']
    probabilities = predictions.flatten()
    thresholds = [threshold] * len(slugs)

    predictions = np.zeros(len(probabilities))
    predictions[probabilities < threshold] = 0
    predictions[probabilities >= threshold] = 1

    predictions = predictions.flatten()

    predictions = np.vectorize(pass_dict.get)(predictions)

    output = pd.DataFrame({'Bill Slug': slugs, 'Probabilitiy': probabilities,
                           'Threshold': thresholds, 'Predictions': predictions})
    output.set_index('Bill Slug', inplace=True)
    output.to_csv(args.output_data)

    print('Output')
    print(output.round(6))
