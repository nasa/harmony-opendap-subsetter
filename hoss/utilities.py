""" This module contains lower-level functionality that can be abstracted into
    functions. Primarily this improves readability of the source code, and
    allows finer-grained unit testing of each smaller part of functionality.

"""
from logging import Logger
from os import sep
from os.path import splitext
from shutil import move
from typing import Dict, Optional, Set, Tuple
from urllib.parse import quote
from uuid import uuid4
import mimetypes

from harmony.exceptions import ForbiddenException, ServerException
from harmony.util import Config, download as util_download

from hoss.exceptions import UrlAccessFailed, UrlAccessFailedWithRetries


HTTP_REQUEST_ATTEMPTS = 3


def get_file_mimetype(file_name: str) -> Tuple[Optional[str], Optional[str]]:
    """ This function tries to infer the MIME type of a file string. If
        the `mimetypes.guess_type` function cannot guess the MIME type of the
        granule, a default value is returned, which assumes that the file is
        a NetCDF-4 file.

    """
    mimetype = mimetypes.guess_type(file_name, False)

    if not mimetype or mimetype[0] is None:
        mimetype = ('application/x-netcdf4', None)

    return mimetype


def get_opendap_nc4(url: str, required_variables: Set[str], output_dir: str,
                    logger: Logger, access_token: str, config: Config) -> str:
    """ Construct a semi-colon separated string of the required variables and
        use as a constraint expression to retrieve those variables from
        OPeNDAP.

        Returns the path of the downloaded granule containing those variables.

    """
    constraint_expression = get_constraint_expression(required_variables)
    netcdf4_url = f'{url}.dap.nc4'

    if constraint_expression != '':
        request_data = {'dap4.ce': constraint_expression}
    else:
        request_data = None

    downloaded_nc4 = download_url(netcdf4_url, output_dir, logger,
                                  access_token=access_token, config=config,
                                  data=request_data)

    # Rename output file, to ensure repeated data downloads to OPeNDAP will be
    # respected by `harmony-service-lib-py`.
    return move_downloaded_nc4(output_dir, downloaded_nc4)


def get_constraint_expression(variables: Set[str]) -> str:
    """ Take a set of variables and return a URL encoded, semi-colon separated
        DAP4 constraint expression to retrieve those variables. Each variable
        may or may not specify their index ranges.

    """
    return quote(';'.join(variables), safe='')


def move_downloaded_nc4(output_dir: str, downloaded_file: str) -> str:
    """ Change the basename of a NetCDF-4 file downloaded from OPeNDAP. The
        `harmony-service-lib-py` produces a local filename that is a hex digest
        of the requested URL only. If this filename is already present in the
        local file system, `harmony-service-lib-py` assumes it does not need to
        make another HTTP request, and just returns the constructed file path,
        even if a POST request is being made with different parameters.

    """
    extension = splitext(downloaded_file)[1] or '.nc4'
    new_filename = sep.join([output_dir, f'{uuid4().hex}{extension}'])
    move(downloaded_file, new_filename)
    return new_filename


def download_url(url: str, destination: str, logger: Logger,
                 access_token: str = None, config: Config = None,
                 data=None) -> str:
    """ Use built-in Harmony functionality to download from a URL. This is
        expected to be used for obtaining the granule `.dmr` and the granule
        itself (only the required variables).

        OPeNDAP can return intermittent 500 errors. This function will retry
        the original request in the event of a 500 error, but not for other
        error types. In those instances, the original HTTPError is re-raised.

        The return value is the location in the file-store of the downloaded
        content from the URL.

    """
    logger.info(f'Downloading: {url}')

    if data is not None:
        logger.info(f'POST request data: "{format_dictionary_string(data)}"')

    request_completed = False
    attempts = 0

    while not request_completed and attempts < HTTP_REQUEST_ATTEMPTS:
        attempts += 1

        try:
            response = util_download(
                url,
                destination,
                logger,
                access_token=access_token,
                data=data,
                cfg=config
            )
            request_completed = True
        except ForbiddenException as harmony_exception:
            raise UrlAccessFailed(url, 400) from harmony_exception
        except ServerException as harmony_exception:
            if attempts < HTTP_REQUEST_ATTEMPTS:
                logger.info('500 error returned, retrying request.')
            else:
                raise UrlAccessFailedWithRetries(url) from harmony_exception
        except Exception as harmony_exception:
            # Not a 500 error, so raise immediately and exit the loop.
            raise UrlAccessFailed(url, 'Unknown') from harmony_exception

    return response


def format_variable_set_string(variable_set: Set[str]) -> str:
    """ Take an input set of variable strings and return a string that does not
        contain curly braces, for compatibility with Harmony logging.

    """
    return ', '.join(variable_set)


def format_dictionary_string(dictionary: Dict) -> str:
    """ Take an input dictionary and return a string that does not contain
        curly braces (assuming the dictionary is not nested, or doesn't contain
        set values).

    """
    return '\n'.join([f'{key}: {value}' for key, value in dictionary.items()])


def get_value_or_default(value: Optional[float], default: float) -> float:
    """ A helper function that will either return the value, if it is supplied,
        or a default value if not.

    """
    return value if value is not None else default


def rgetattr(input_object, requested_attribute, *args):
    """ This is a recursive version of the inbuilt `getattr` method, such that
        it can be called to retrieve nested attributes. For example:
        the Message.subset.shape within the input Harmony message.

        Note, if a default value is specified, this will be returned if any
        attribute in the specified chain is absent from the supplied object.

    """
    if '.' not in requested_attribute:
        result = getattr(input_object, requested_attribute, *args)
    else:
        attribute_pieces = requested_attribute.split('.')
        result = rgetattr(getattr(input_object, attribute_pieces[0], *args),
                          '.'.join(attribute_pieces[1:]), *args)

    return result
