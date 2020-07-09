""" Utility classes used to extend the unittest capabilities """
import re

from requests import HTTPError


class MockResponse:
    """ A test class to be used in mocking a response from the `requests.get`
        method.

    """
    def __init__(self, status_code: int, content: str):
        self.status_code = status_code
        self.content = content

    def raise_for_status(self):
        """ Check the response status code. If it isn't in the range of
            expected successful status codes, then raise an exception.

        """
        if self.status_code > 299 or self.status_code < 200:
            raise HTTPError('Could not retrieve data.')


class contains(str):
    """ Extension class that allows a 'string contains' check in a unit test
        assertion, e.g.: x.assert_called_once_with(contains('string content'))

    """
    def __eq__(self, other):
        return self.lower() in other.lower()


class matches(str):
    """ Extentsion class that allows a regular expression type check in a unit
        test assertion, e.g.: x.assert_called_once_with(matches(regex))

    """
    def __eq__(self, other):
        return re.search(self.lower(), other.lower(), re.IGNORECASE)
