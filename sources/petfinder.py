"""Module for collecting data from petfinder.com"""

import os
import requests
from typing import List
from csv import DictWriter

# Package modules
from utils import config, DATA_PATH, TIMESTAMP


class PetFinderApi(object):
    """Fetch pet data from petfinder.com

    Attributes:
        _TOKEN_URL: URL endpoint for authenticating token access.
        _ANIMALS_URL: URL endpoint for fetching pet animal data.
        _access_token: Stores retrieved access token
        _SOURCE: Identify for this api
        _fieldnames: Stores unique fieldnames for parsed data.
    """
    def __init__(self):
        self._TOKEN_URL = "https://api.petfinder.com/v2/oauth2/token"
        self._ANIMALS_URL = 'https://api.petfinder.com/v2/animals'
        self._access_token = None
        self._SOURCE = 'petfinder'
        self._fieldnames = set()

    def _get_access_token(self) -> dict:
        """Authenticate access to API endpoints.

        Returns:
            A dict of the response that includes token_type, expires_in,
            and access_token.
        """
        params = config(section='petfinder')

        payload = {
            'grant_type': 'client_credentials',
            'client_id': params['client_id'],
            'client_secret': params['client_secret']
        }

        headers = {
            'Content-Type': "application/x-www-form-urlencoded",
            'cache-control': "no-cache"
        }

        response = requests.post(self._TOKEN_URL, data=payload, headers=headers)
        # TODO: logic for handling failed requests
        return response.json()

    def _fetch_animals(self, location='texas', page=1) -> dict:
        """Fetch animals data for specified location and page number."""
        if self._access_token is None:
            self._access_token = self._get_access_token()

        headers = {
            'Authorization': f'{self._access_token["token_type"]} {self._access_token["access_token"]}',
            'cache-control': "no-cache"
        }
        qs = {
            'limit': 100,
            'location': location,
            'page': page
        }
        response = requests.get(self._ANIMALS_URL, headers=headers,
                                params=qs)
        # TODO: logic for handling failed requests
        return response.json()

    def get_all_animals(self, location='texas') -> List[dict]:
        animals = list()

        result = self._fetch_animals(location=location)
        animals.extend(result['animals'])
        total_count = int(result['pagination']['total_count'])
        total_pages = int(result['pagination']['total_pages'])

        # Get remaining pages
        for i in range(2, total_pages + 1):
            result = self._fetch_animals(location=location,
                                         page=i)
            animals.extend(result['animals'])

        assert len(animals) == total_count
        return animals

    def munge_animal_data(self, animals: List[dict]) -> List[dict]:
        """Process animal response data into storable format and schema.

        The response data is list of dictionaries. Some dictionary keys
        have dictionary values. Need to hoist lower level dictionary
        data values up to top level or omit.
        """
        # List of fields to exclude for storage
        omit = ['tags', 'photos', 'contact', '_links']

        munged = list()

        # Iterate over each animal data
        for animal_dict in animals:
            data = dict()
            # Iterate over fields in animals dict
            for field in animal_dict:
                if field in omit:
                    continue
                value = animal_dict[field]

                # Parse dict values into higher level keys
                if isinstance(value, dict):
                    for key in value:
                        field_name = f'{field}_{key}'
                        data[field_name] = value[key]
                        self._fieldnames.add(field_name)
                else:
                    data[field] = value
                    self._fieldnames.add(field)
            munged.append(data)

        return munged

    def write_to_csv(self, data: List[dict], location='texas') -> str:
        fname = f'{self._SOURCE}_{location}_{TIMESTAMP}.csv'
        path = os.path.join(DATA_PATH, fname)

        with open(file=path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = DictWriter(csvfile, fieldnames=self._fieldnames)
            writer.writeheader()

            for animal in data:
                writer.writerow(animal)

        return path


if __name__ == '__main__':
    api = PetFinderApi()
    result = api.get_all_animals()
    data = api.munge_animal_data(animals=result)
    path = api.write_to_csv(data)
    print('Flat file saved at:', path)
