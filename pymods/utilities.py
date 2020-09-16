""" This module contains lower-level functionality that can be abstracted into
    functions. Primarily this improves readability of the source code, and
    allows finer-grained unit testing of each smaller part of functionality.

"""
from logging import Logger
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError
from xml.etree.ElementTree import Element
import functools
import mimetypes
import re

from harmony.util import download as util_download
import numpy as np

from pymods.exceptions import DmrNamespaceError, UrlAccessFailedWithRetries


DAP4_TO_NUMPY_MAP = {'Char': np.uint8, 'Byte': np.uint8, 'Int8': np.int8,
                     'UInt8': np.uint8, 'Int16': np.int16, 'UInt16': np.uint16,
                     'Int32': np.int32, 'UInt32': np.uint32, 'Int64': np.int64,
                     'UInt64': np.uint64, 'Float32': np.float32,
                     'Float64': np.float64, 'String': str, 'URL': str,
                     'Dimension': None}

HTTP_REQUEST_ATTEMPTS = 3


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


def download_url(url: str, destination: str, logger: Logger) -> str:
    """ Use built-in Harmony functionality to download from a URL. This is
        expected to be used for obtaining the granule `.dmr` and the granule
        itself (only the required variables).

        OPeNDAP can return intermittent 500 errors. This function will retry
        the original request in the event of a 500 error, but not for other
        error types. In those instances, the original HTTPError is re-raised.

        The return value is the location in the file-store of the downloaded
        content from the URL.

    """
    request_completed = False
    attempts = 0

    while not request_completed and attempts < HTTP_REQUEST_ATTEMPTS:
        attempts += 1

        try:
            response = util_download(url, destination, logger)
            request_completed = True
        except HTTPError as http_exception:
            if http_exception.code == 500 and attempts < HTTP_REQUEST_ATTEMPTS:
                logger.debug('500 error returned, retrying request.')
            elif http_exception.code == 500:
                raise UrlAccessFailedWithRetries(url)
            else:
                # Not a 500 error, so re-raise the HTTPError and exit the loop.
                # (Also re-raise if this is the final attempt)
                raise http_exception

    return response
