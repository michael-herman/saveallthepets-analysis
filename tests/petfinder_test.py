import unittest
import os

# Package modules
from sources.petfinder import PetFinderApi


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._api = PetFinderApi()

    def test_get_access_token(self):
        """Test get_access_token method"""
        expected_keys = [
            'token_type', 'expires_in', 'access_token'
        ]

        response = self._api._get_access_token()
        # Verify response contain expected keys
        for key in expected_keys:
            self.assertTrue(key in response,
                            msg=f'{key} not in response: {response.keys()}')

        # Verify token_type = Bearer
        self.assertEqual('Bearer', response['token_type'],
                         msg=f'Incorrect token_type value: {response["token_type"]}')

    def test_fetch_animals(self):
        """Test for get_animals method"""
        response = self._api._fetch_animals()
        for key in ['animals', 'pagination']:
            self.assertTrue(key in response,
                            msg=f'{key} not in {response.keys()}')

    def test_munge_animal_data(self):
        """Test munge_animal_data method"""
        result = self._api._fetch_animals()
        animals = result['animals']
        data = self._api.munge_animal_data(animals)

        for animal in data:
            # Case 1: verify excluded fields
            for field in ['tags', 'photos', 'contact', '_links']:
                self.assertTrue(field not in animal,
                                msg=f'{field} is in {animal.keys()}')
            # Case 2: verify no field values are dict type
            for value in animal.values():
                self.assertFalse(isinstance(value, dict),
                                 msg=f'{value} is dict type: {value}')

    def test_write_to_csv(self):
        """Test for write_to_csv method."""
        result = self._api._fetch_animals()
        animals = result['animals']
        data = self._api.munge_animal_data(animals)

        path = self._api.write_to_csv(data)
        self.assertTrue(os.path.isfile(path),
                        msg=f'Invalid file: {path}')


if __name__ == '__main__':
    unittest.main()
