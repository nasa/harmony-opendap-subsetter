""" This module contains lower-level functionality that can be abstracted into
    functions. Primarily this improves readability of the source code, and
    allows finer-grained unit testing of each smaller part of functionality.

"""
import os
from logging import Logger
from typing import Dict, List, Optional, Tuple
import functools
import mimetypes
import requests

from pymods.exceptions import PydapRetrievalError, AuthorizationError


def get_url_response(source_url: str, logger: Logger):
    """ This function gets response from source url request"""

    username = os.environ.get('EDL_USERNAME')
    password = os.environ.get('EDL_PASSWORD')
    if (username is None or password is None):
        logger.error('There are no EDL_USERNAME and/or EDL_PASSWORD in the system environment')
        raise AuthorizationError(
            "There are no EDL_USERNAME and/or EDL_PASSWORD in the system environment")
    try:
        response = requests.get(source_url, auth=(username, password))
        response.raise_for_status()
    except requests.HTTPError:
        logger.error('Request cannot be completed with error code '
                     f'{response.status_code}')
        raise requests.HTTPError('Request cannot be completed with error code '
                                 f'{response.status_code}')
    return response

def get_file_mimetype(file_name: str) -> Tuple[Optional[str]]:
    """ This function tries to infer the MIME type of a file string. If
        the `mimetypes.guess_type` function cannot guess the MIME type of the
        granule, a default value is returned.

    """
    mimetype = mimetypes.guess_type(file_name, False)

    if not mimetype or mimetype[0] is None:
        mimetype = ('application/octet-stream', None)

    return mimetype

def recursive_get(input_dictionary: Dict, keys: List[str]):
    """ Extract a value from an aribtrarily nested dictionary. """
    try:
        nested_value = functools.reduce(dict.get, keys, input_dictionary)
    except TypeError:
        # This catches when there is a missing intermediate key
        nested_value = None

    return nested_value


def pydap_attribute_path(full_path: str) -> List[str]:
    """ Take the full path to a metadata attribute and return the list of
        keys that locate that attribute within the pydap global attributes.
        This function expects the input path to have a leading "/" character.

    """
    full_path_pieces = full_path.split('/')[1:]

    final_key = full_path_pieces.pop(-1)
    leading_key = '_'.join(full_path_pieces)

    if leading_key:
        key_list = [leading_key, final_key]
    else:
        key_list = [final_key]

    return key_list
