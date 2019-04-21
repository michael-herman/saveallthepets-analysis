"""Module for collecting data from petfinder.com"""

import requests
from utils import config


class PetFinderApi:
    def __init__(self):
        self._TOKEN_URL = "https://api.petfinder.com/v2/oauth2/token"
        self._ACCESS_TOKEN = None

    def _get_access_token(self) -> dict:
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

