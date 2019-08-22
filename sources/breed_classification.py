"""Module for pulling breed information from Microsoft"""

import os
import requests
import pandas as pd
from typing import List
from csv import DictWriter
from bs4 import BeautifulSoup

class BreedClassification(object):
    """Fetch breed data from Google

    Attributes:
        ...
    """
    def __init__(self):
        # self._TOKEN_URL = "https://api.petfinder.com/v2/oauth2/token"
        # self._ANIMALS_URL = 'https://api.petfinder.com/v2/animals'
        # self._access_token = None
        # self._SOURCE = 'petfinder'
        # self._fieldnames = set()

    def _get_access_token(self) -> dict:
        """Authenticate access to API endpoints.

        Returns:
            A dict of the response that includes token_type, expires_in,
            and access_token.
        """
        pass
