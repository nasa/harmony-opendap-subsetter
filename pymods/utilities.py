""" This module contains lower-level functionality that can be abstracted into
    functions. Primarily this improves readability of the source code, and
    allows finer-grained unit testing of each smaller part of functionality.

"""
from typing import Dict, List, Optional, Tuple
from urllib import request, error
import functools
import json
import mimetypes
import os
import re
from logging import Logger

from harmony.message import Granule


def get_granule_mimetype(granule: Granule) -> Tuple[Optional[str]]:
    """ This function tries to infer the MIME type of the input granule. If
        the `mimetypes.guess_type` function cannot guess the MIME type of the
        granule, a default value is returned.

    """
    mimetype = mimetypes.guess_type(granule.local_filename, False)

    if not mimetype or mimetype[0] is None:
        mimetype = ('application/octet-stream', None)

    return mimetype

def get_token(logger: Logger) -> str:
    """ This function creates token for CMR query using EDL_USERNAME and EDL_password.
    """
    # get collection EntryTitle from CMR query
    username = os.environ['EDL_USERNAME']
    password = os.environ['EDL_PASSWORD']

    token_generator = f'<token><username>{username}</username>' \
                      f'<password>{password}</password>' \
                      f'<client_id>Var_Subsetter</client_id>' \
                      f'<user_ip_address>127.0.0.1</user_ip_address> </token>'
    token_url = 'https://cmr.uat.earthdata.nasa.gov/legacy-services/rest/tokens/'
    headers = {'Content-Type': 'application/xml'}

    token = ''

    try:
        req = request.Request(token_url, data=token_generator.encode('utf-8'),
                              method='POST', headers=headers)
        res = request.urlopen(req)
    except error.HTTPError:
        logger.error("Failed to generate CMR token")

    else:
        data = res.read().decode('utf-8')
        token = re.search('<id>(.*)</id>', data).group(1)

    return token

def cmr_query(query_type: str, concept_id: str, query_item: str, token: str, logger: Logger) -> str:
    """ CMR query
    """
    cmr_url = f'https://cmr.uat.earthdata.nasa.gov/search/{query_type}.umm_json_v1_4?' \
              f'concept_id={concept_id}&token={token}'

    result = ''

    with request.urlopen(cmr_url) as url:
        data = json.loads(url.read().decode())
        try:
            result = data['items'][0]['umm'][query_item]
        except IndexError:
            logger.error(f'Unable to obtain {query_item} from CMR')

    return result


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
