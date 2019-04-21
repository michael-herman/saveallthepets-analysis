import unittest
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
                            msg=f'{key} not in response: {response}')

        # Verify token_type = Bearer
        self.assertEqual('Bearer', response['token_type'],
                         msg=f'Incorrect token_type value: {response["token_type"]}')


if __name__ == '__main__':
    unittest.main()
