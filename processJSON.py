'''
Takes raw congress data and creates a csv of the dataset for ml
'''
import json
import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm
import requests


def get_args():
    '''
    Gets args for this program

    param:
    None

    return:
    parsed arguments object
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', default='data/',
                        help='Relative data path to main data folder', type=str)
    parser.add_argument('-cong_data', default='data/117/members.json',
                        help='Relative data path to json data for congress member characteristics', type=str)

    return(parser.parse_args())


def walk_dir(input_path):
    '''
    Walks through subdirectories to grab all necessary data

    param:
    input_path - str path to parent folder containing all data

    return:
    path - str path to file containing data
    '''
    for subdir, dirs, files in os.walk(input_path):
        for name in files:
            if(name == 'data.json' or name == 'characteristics.json'):
                yield os.path.abspath(os.path.join(subdir, name))


def get_member_party(member_data, members, cong_json):
    '''
    Return member party from a given member's json data

    param:
    member_data - json data representing member characteristics
    members - dict for member parties
    cong_json - dict for congress json data for member characteristics

    return:
    party - str for member party, either D, R, or I
    members - dict for member parties updated
    '''
    f = open('api_key.json' ,'r')
    json_data = json.load(f)
    api_key = json_data['key']

    try:
        id_type = 'bioguide'
        id = member_data[id_type + '_id']
    except:
        id_type = 'thomas'
        id = member_data[id_type + '_id']

    try:
        sponsor_party = members[id]
    except:
        for person in cong_json:
            try:
                if(person['id'] == id):
                    member = person
                    break
            except:
                pass
            
        try:
            member_terms = [term['party'] for term in member['terms']]
            (values, counts) = np.unique(
                member_terms, return_counts=True)
            indx = np.argmax(counts)
            sponsor_party = values[indx][0]
            members[id] = sponsor_party
        except KeyError: # To support old and new api formats
            members[id] = sponsor_party = member['party']

    return sponsor_party, members


def get_bill_char(bill_json, members, cong_json, chamber, congress_chars, get_status=True):
    '''
    Gets bill characteristics from json data

    param:
    bill_json - dict json data for bill
    members - dict of member's party afiliation
    cong_json - dict for congress json data for member characteristics
    chamber - str indicating chamber : house or senate
    congress_chars - dict of congress party makeup
    get_status - if the status should be returned, in other words is the bill labeled

    return:
    output - list of bill characteristics
    chamber - str indicating house or senate
    '''
    bill_subj = ','.join(bill_json['subjects'])
    num_cosponsors = 0
    bipartisan = False

    sponsor_party, members = get_member_party(
        bill_json['sponsor'], members, cong_json)

    sponsor_states = [bill_json['sponsor']['state']]

    for cosponsor in bill_json['cosponsors']:

        num_cosponsors += 1

        cosponsor_party, members = get_member_party(
            cosponsor, members, cong_json)

        if(cosponsor_party != sponsor_party):
            bipartisan = True

        if(cosponsor['state'] not in sponsor_states):
            sponsor_states.append(cosponsor['state'])

    if(get_status):
        outcome = bill_json['status']
    else:
        outcome = None

    output = [bill_subj, sponsor_party,
              bipartisan, num_cosponsors, len(sponsor_states), outcome]

    # Don't include entries with missing data
    if((None not in output) or (not get_status)):
        # Find bill name from path
        if(chamber == 'house'):
            output.insert(
                0, congress_chars['House']['Independent'])
            output.insert(0, congress_chars['House']['Republican'])
            output.insert(0, congress_chars['House']['Democrat'])

        elif(chamber == 'senate'):
            output.insert(
                0, congress_chars['Senate']['Independent'])
            output.insert(
                0, congress_chars['Senate']['Republican'])
            output.insert(0, congress_chars['Senate']['Democrat'])
    else:
        output = None
    if(bill_json['bill_slug'] == 'hr5683'):
        print(output)
    return output


if __name__ == '__main__':
    args = get_args()
    data_path = args.data

    # get total num files for progress bar
    filecounter = 0
    for filepath in walk_dir(data_path):
        filecounter += 1

    congress_chars = {}

    # lists for running through data
    house_list = []
    senate_list = []

    # dict to cache the members' party
    members = {}

    f = open(args.cong_data)
    cong_json = json.load(f)

    for filepath in tqdm(walk_dir(data_path), total=filecounter, unit="files"):
        try:
            f = open(filepath)
            json_data = json.load(f)

            if(('data.json' in filepath) and ('amendments' not in filepath)):
                filepath = filepath.split('/')
                indx = filepath.index('data.json')
                bill_id = filepath[indx - 1]

                if(bill_id[0] == 'h'):
                    chamber = 'house'
                elif(bill_id[0] == 's'):
                    chamber = 'senate'

                output = get_bill_char(
                    json_data, members, cong_json, chamber, congress_chars)

                if(chamber == 'house'):
                    house_list.append(output)

                elif(chamber == 'senate'):
                    senate_list.append(output)

            elif('characteristics.json' in filepath):
                congress_chars = json_data
        except:
            pass

    column_names = ['Democrat', 'Republican', 'Independant',
                    'Subjects', 'Sponsor Party', 'Bipartisan', 'Number Cosponsors', 'Number Cosponsor States', 'Outcome']

    house = pd.DataFrame(house_list, columns=column_names)
    senate = pd.DataFrame(senate_list, columns=column_names)

    house.to_csv(os.path.join(data_path, 'house.csv'),
                 index=False, header=True)
    senate.to_csv(os.path.join(data_path, 'senate.csv'),
                  index=False, header=True)
