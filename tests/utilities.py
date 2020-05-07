""" Utility classes used to extend the unittest capabilities """
import re


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
