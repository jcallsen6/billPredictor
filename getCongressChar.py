'''
Gets the number of each party for each chamber of congress and saves the results
to a characteristics.json in each congress instance's subfolder
Gets total members which doesn't account for members dropping out and new ones filling their position
Fixed with https://en.wikipedia.org/wiki/Party_divisions_of_United_States_Congresses
'''

import requests
import json
import sys
import argparse
from tqdm import tqdm


def get_args():
    '''
    Gets args for this program

    param:
    None

    return:
    parsed arguments object
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-instances', default=(113, 116),
                        help='Range of Congresses to get data for', type=int, nargs=2)
    parser.add_argument('-data', default='data/',
                        help='Relative data path to main data folder', type=str)
    parser.add_argument('-api_key', default='api_key.json',
                        help='Relative path to json file containing api key', type=str)
    return(parser.parse_args())


if __name__ == '__main__':
    args = get_args()

    lower_bound = args.instances[0]
    upper_bound = args.instances[1]
    data_folder = args.data
    # make it a directory if not passed correctly
    if(data_folder[-1] != '/'):
        data_folder = data_folder + '/'

    url = 'https://api.propublica.org/congress/v1/{0}/{1}/members.json'

    with open(args.api_key, 'rb') as api_file:
        api_key = json.load(api_file)
        api_key = api_key['key']

    for congress in tqdm(range(lower_bound, upper_bound+1)):
        characteristics = {'House': {}, 'Senate': {}}
        characteristics['House'] = {'Democrat': 0,
                                    'Republican': 0, 'Independent': 0}
        characteristics['Senate'] = {
            'Democrat': 0, 'Republican': 0, 'Independent': 0}

        for chamber in characteristics.keys():
            r = requests.get(url.format(str(congress), str(chamber)),
                             headers={'X-API-Key': api_key})
            members = r.json()['results'][0]['members']

            for mem in members:
                if(mem['party'] == 'D'):
                    characteristics[chamber]['Democrat'] += 1
                elif(mem['party'] == 'R'):
                    characteristics[chamber]['Republican'] += 1
                else:
                    characteristics[chamber]['Independent'] += 1

        with open(data_folder + str(congress) + '/characteristics.json', 'w') as output:
            json.dump(characteristics, output)
