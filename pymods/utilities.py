""" This module contains lower-level functionality that can be abstracted into
    functions. Primarily this improves readability of the source code, and
    allows finer-grained unit testing of each smaller part of functionality.

"""
from logging import Logger
from typing import Dict, List, Optional, Tuple
from xml.etree.ElementTree import Element
import functools
import mimetypes
import os
import re

import numpy as np
import requests

from pymods.exceptions import AuthorizationError, DmrNamespaceError


DAP4_TO_NUMPY_MAP = {'Char': np.uint8, 'Byte': np.uint8, 'Int8': np.int8,
                     'UInt8': np.uint8, 'Int16': np.int16, 'UInt16': np.uint16,
                     'Int32': np.int32, 'UInt32': np.uint32, 'Int64': np.int64,
                     'UInt64': np.uint64, 'Float32': np.float32,
                     'Float64': np.float64, 'String': str, 'URL': str}

def get_url_response(source_url: str, logger: Logger) -> requests.models.Response:
    """ This function gets response from source url request, if the HTTP
        response contains a non-200 status code, the function will raise an
        exception.

    """
    try:
        response = requests.get(source_url)
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
    full_path_pieces = full_path.lstrip('/').split('/')

    final_key = full_path_pieces.pop(-1)
    leading_key = '_'.join(full_path_pieces)

    if leading_key:
        key_list = [leading_key, final_key]
    else:
        key_list = [final_key]

    return key_list


def get_xml_namespace(root_element: Element) -> str:
    """ Given the root element of an XML document, extract the associated
        namespace. This allows for the full qualification of child elements.
        The root element of a dmr file is expected to be a Dataset tag.

    """
    match = re.match('(.+)Dataset', root_element.tag)

    if match:
        xml_namespace = match.groups()[0]
    else:
        raise DmrNamespaceError(root_element.tag)

    return xml_namespace


def get_xml_attribute(variable: Element, attribute_name: str, namespace: str,
                      default_value: Optional = None) -> Optional:
    """ Extract the value of an XML Attribute tag from a `.dmr`. First search
        the supplied variable element for a fully qualified Attribute child
        element, with a name property matching the requested attribute name. If
        there is no matching tag, return the `default_value`, which can be
        user-defined, or default to `None`. If present, the returned value is
        cast as the type indicated by the Attribute tag's `type` property.

    """
    attribute_element = variable.find(f'{namespace}Attribute'
                                      f'[@name="{attribute_name}"]')

    if attribute_element is not None:
        value_type = attribute_element.get('type', 'String')
        value_element = attribute_element.find(f'{namespace}Value')

        if value_element is not None:
            numpy_type = DAP4_TO_NUMPY_MAP.get(value_type, str)
            attribute_value = numpy_type(value_element.text)
        else:
            attribute_value = default_value

    else:
        attribute_value = default_value

    return attribute_value


def create_netrc_file(logger: Logger):
    """ Create a .netrc file in the home directory of the Docker container
        using the environment variables supplied to the container by Harmony,
        when it is executed. This file will only be created if a .netrc file
        is not already present on the file system, ensuring local testing,
        outside of Docker, of the variable subsetter will not overwrite
        pre-existing .netrc files.

        This file will be automatically detected by the `requests.get` function
        when making calls to OPeNDAP.

    """
    edl_username = os.environ.get('EDL_USERNAME')
    edl_password = os.environ.get('EDL_PASSWORD')

    if edl_username is None or edl_password is None:
        raise AuthorizationError('EDL_USERNAME and/or EDL_PASSWORD missing '
                                 'from the system environment.')

    urs_prefixes = ['', 'uat.', 'sit.']
    urs_machines = [f'{prefix}urs.earthdata.nasa.gov'
                    for prefix in urs_prefixes]

    netrc_path = os.sep.join([os.path.expanduser('~'), '.netrc'])

    if not os.path.isfile(netrc_path):
        logger.info(f'Creating .netrc for user: {edl_username}')
        with open(netrc_path, 'w')as file_handler:
            for machine in urs_machines:
                file_handler.writelines([f'machine {machine}\n',
                                         f'    login {edl_username}\n',
                                         f'    password {edl_password}\n',
                                         '\n'])
