""" This module contains lower-level functionality that can be abstracted into
    functions. Primarily this improves readability of the source code, and
    allows finer-grained unit testing of each smaller part of functionality.

"""
from typing import Optional, Tuple
import mimetypes
from urllib import request, error
import os
import re
import json
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
