""" Utility classes used to extend the unittest capabilities """
import re

from requests import HTTPError


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


def write_dmr(output_dir: str, content: str):
    """ A helper function to write out the content of a `.dmr`, when the
        `harmony.util.download` function is called. This will be called as
        a side-effect to the mock for that function.

    """
    dmr_name = f'{output_dir}/downloaded.dmr'

    with open(dmr_name, 'w') as file_handler:
        file_handler.write(content)

    return dmr_name
