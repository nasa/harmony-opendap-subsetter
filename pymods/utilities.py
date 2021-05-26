""" This module contains lower-level functionality that can be abstracted into
    functions. Primarily this improves readability of the source code, and
    allows finer-grained unit testing of each smaller part of functionality.

"""
from logging import Logger
from os import sep
from os.path import splitext
from shutil import copy
from typing import Optional, Set, Tuple
from urllib.error import HTTPError
from urllib.parse import quote
from uuid import uuid4
import mimetypes

from harmony.util import Config, download as util_download
import numpy as np

from pymods.exceptions import UrlAccessFailed, UrlAccessFailedWithRetries


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
    request_data = {'dap4.ce': constraint_expression}

    downloaded_nc4 = download_url(f'{url}.dap.nc4', output_dir, logger,
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
    extension = splitext(downloaded_file)[1]
    new_filename = sep.join([output_dir, f'{uuid4().hex}{extension}'])
    copy(downloaded_file, new_filename)
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
        logger.info(f'POST request data: "{data}"')

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
        except HTTPError as http_exception:
            if http_exception.code == 500 and attempts < HTTP_REQUEST_ATTEMPTS:
                logger.info('500 error returned, retrying request.')
            elif http_exception.code == 500:
                raise UrlAccessFailedWithRetries(url) from http_exception
            else:
                # Not a 500 error, so raise immediately and exit the loop.
                raise UrlAccessFailed(url, http_exception.code) from http_exception

    return response
