#!/usr/bin/env python
# coding: utf-8

import json
import requests
import pickle
import pandas as pd
import numpy as np
from processJSON import get_bill_char
from predictChamber import encode_data_pred
from keras.models import load_model
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

    parser.add_argument('-api_file', default='api_key.json',
                        help='Relative path to api json file', type=str)
    parser.add_argument('-house_model_name', default='models/houseModel',
                        help='Relative path to house model folder', type=str)
    parser.add_argument('-senate_model_name', default='models/senateModel',
                        help='Relative path to senate model folder', type=str)
    parser.add_argument('-session', default=117,
                        help='Integer session of congress', type=str)
    parser.add_argument('-output', default='curr_pred.json',
                        help='Relative path to json output for frontend')

    return(parser.parse_args())


def preprocess_data(bill, session, members, member_char, chamber, cong_char, api_key):
    '''
    Preprocesses bill from current bill call to fit the form for encoding
    
    param:
    bill - json of bill from current bill api call
    session - int session of congress ie 117
    members - dict of member parties
    member_char - csv of all member parties
    chamber - str name of chamber either house or senate
    congress_char - dict for congress json data for member characteristics
    api_key - string api key for prorepublica
    
    return:
    list of bill characteristics for prediction
    '''
    bill_url = 'https://api.propublica.org/congress/v1/{0}/bills/{1}/{2}.json' 
    bill_slug = bill['bill_slug']
    response = requests.get(bill_url.format(str(session), bill_slug, 'subjects'), headers={'X-API-KEY' : api_key})
    
    bill['subjects'] = [entry['name'] for entry in response.json()['results'][0]['subjects']]
    bill['sponsor'] = {'bioguide_id' : response.json()['results'][0]['sponsor_id'], 'state' : response.json()['results'][0]['sponsor_state']}
    
    response = requests.get(bill_url.format(str(session), bill_slug, 'cosponsors'), headers={'X-API-KEY' : api_key})
    cosponsors = response.json()['results'][0]['cosponsors']
    for cosponsor in cosponsors:
        cosponsor['bioguide_id'] = cosponsor['cosponsor_id']
        cosponsor['state'] = cosponsor['cosponsor_state']
 
    bill['cosponsors'] = cosponsors
    
    return(get_bill_char(bill, members, member_char, chamber, cong_char, get_status=False))


def main():
    args = get_args()

    with open(args.house_model_name + '/params', 'rb') as params_file:
        house_dict = pickle.load(params_file)
        house_threshold = house_dict['thresh']
        house_party_indicies = house_dict['party_indicies']
        max_len = house_dict['max_len']
    with open(args.senate_model_name + '/params', 'rb') as params_file:
        senate_dict = pickle.load(params_file)
        senate_threshold = senate_dict['thresh']
        senate_party_indicies = house_dict['party_indicies']

    member_char = f"data/{args.session}/members.json"

    url = 'https://api.propublica.org/congress/v1/{0}/{1}/bills/active.json'
    f = open(args.api_file ,'r')
    json_data = json.load(f)
    api_key = json_data['key']
    
    char_f = open(f"data/{args.session}/characteristics.json", 'r')
    cong_char = json.load(char_f)

    recent_house = requests.get(url.format(args.session, 'house'), headers={'X-API-Key': api_key})
    house_bills = recent_house.json()['results'][0]['bills']

    recent_senate = requests.get(url.format(args.session, 'senate'), headers={'X-API-Key': api_key})
    senate_bills = recent_senate.json()['results'][0]['bills']

    house_output = []
    senate_output = []
    members = {}
    f = open(member_char, 'r')
    member_char = json.load(f)

    for bill in house_bills:
        house_output.append(preprocess_data(bill, args.session, members, member_char, 'house', cong_char, api_key))
        
    for bill in senate_bills:
        senate_output.append(preprocess_data(bill, args.session, members, member_char, 'senate', cong_char, api_key))

    column_names = ['Democrat', 'Republican', 'Independant',
                    'Subjects', 'Sponsor Party', 'Bipartisan', 'Number Cosponsors', 'Number Cosponsor States', 'Outcome']

    house = pd.DataFrame(house_output, columns=column_names)
    house_subjects = house['Subjects']
    senate = pd.DataFrame(senate_output, columns=column_names)
    senate_subjects = senate['Subjects']

    house_char_features, house_text_features = encode_data_pred(house, max_len, house_party_indicies, args.house_model_name)
    senate_char_features, senate_text_features = encode_data_pred(senate, max_len, senate_party_indicies, args.house_model_name)
    print(house_char_features)
    house_char_features = house_char_features.astype(np.float32)
    senate_char_features = senate_char_features.astype(np.float32)

    house_model = load_model(args.house_model_name + '/model.h5')
    senate_model = load_model(args.senate_model_name + '/model.h5')

    house_predictions = house_model.predict([house_char_features, house_text_features])
    senate_predictions = senate_model.predict([senate_char_features, senate_text_features])


    house_slugs = [entry['bill_slug'] for entry in house_bills]
    senate_slugs = [entry['bill_slug'] for entry in senate_bills]

    house_prob = house_predictions.flatten()
    senate_prob = senate_predictions.flatten()

    pass_dict = {0 : 'Fail', 1 : 'Pass'}


    house_thresholds = [house_threshold] * len(house_slugs)
    senate_thresholds = [senate_threshold] * len(senate_slugs)
    final_house_predictions = np.zeros(len(house_prob))
    final_senate_predictions = np.zeros(len(senate_prob))

    final_house_predictions[house_prob < house_threshold] = 0
    final_house_predictions[house_prob >= house_threshold] = 1
    final_senate_predictions[senate_prob < senate_threshold] = 0
    final_senate_predictions[senate_prob >= senate_threshold] = 1

    final_house_predictions = final_house_predictions.flatten()
    final_senate_predictions = final_senate_predictions.flatten()

    final_house_predictions = np.vectorize(pass_dict.get)(final_house_predictions)
    final_senate_predictions = np.vectorize(pass_dict.get)(final_senate_predictions)

    pred_house_df = pd.DataFrame({'Bill Slug': house_slugs, 'Probability of Passing' : house_prob, 'Model Threshold' : house_threshold, 'Prediction' : final_house_predictions})
    pred_senate_df = pd.DataFrame({'Bill Slug': senate_slugs, 'Probability of Passing' : senate_prob, 'Model Threshold' : senate_threshold, 'Prediction' : final_senate_predictions})

    pred_house_df.set_index('Bill Slug', inplace=True)
    pred_senate_df.set_index('Bill Slug', inplace=True)
    pred_house_df['Probability of Passing'].round(7)
    pred_senate_df['Probability of Passing'].round(7)

    print('House')
    print(pred_house_df.round(6))
    print()
    print('Senate')
    print(pred_senate_df.round(6))
    
    output = {}
    output['house'] = {}
    output['senate'] = {}

    categories = ['Democrat', 'Republican', 'Independant',
                    'Sponsor Party', 'Bipartisan', 'Number Cosponsors', 'Number Cosponsor States']

    for bill_index in range(len(house_char_features)):
        output['house'][bill_index] = {}
        output['senate'][bill_index] = {}
        for index in range(len(categories)):
            output['house'][bill_index][categories[index]] = str(house_char_features[bill_index][index])
            output['senate'][bill_index][categories[index]] = str(senate_char_features[bill_index][index])
            # TODO check number state cosponsors vs num cosponsors

        output['house'][bill_index]['slug'] = house_slugs[bill_index]
        output['house'][bill_index]['subjects'] = house_subjects[bill_index]
        output['house'][bill_index]['raw_prediction'] = str(house_prob[bill_index])
        output['house'][bill_index]['final_prediction'] = str(final_house_predictions[bill_index])

        output['senate'][bill_index]['slug'] = senate_slugs[bill_index]
        output['senate'][bill_index]['subjects'] = senate_subjects[bill_index]
        output['senate'][bill_index]['raw_prediction'] = str(senate_prob[bill_index])
        output['senate'][bill_index]['final_prediction'] = str(final_senate_predictions[bill_index])
    
    with open(args.output, 'w') as outfile:
        json.dump(output, outfile)


if __name__ == '__main__':
    main()


