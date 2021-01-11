""" Utility classes used to extend the unittest capabilities """
import re
from unittest.mock import MagicMock


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


def spy_on(method):
    """
    Creates a spy for the given object instance method which records results
    and return values while letting the call run as normal.  Calls are recorded
    on `spy_on(A.b).mock` (MagicMock), return values are appended to the
    array `spy_on(A.b).return_values`, and exceptions are appended to the array
    `spy_on(A.b).errors`

    The return value should be passed as the third argument to patch.object in
    order to begin recording calls

    Parameters
    ----------
    method : function
        The method to spy on

    Returns
    -------
    function
        A wrapper function that can be passed to patch.object to record calls
    """
    mock = MagicMock()
    return_values = []
    errors = []

    def wrapper(self, *args, **kwargs):
        mock(*args, **kwargs)
        try:
            result = method(self, *args, **kwargs)
        except Exception as err:
            errors.append(err)
            raise
        return_values.append(result)
        return result
    wrapper.mock = mock
    wrapper.return_values = return_values
    wrapper.errors = errors
    return wrapper
