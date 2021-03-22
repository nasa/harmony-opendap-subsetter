""" This module contains lower-level functionality that can be abstracted into
    functions. Primarily this improves readability of the source code, and
    allows finer-grained unit testing of each smaller part of functionality.

"""
from logging import Logger
from typing import Optional, Tuple
from urllib.error import HTTPError
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


def download_url(
        url: str,
        destination: str,
        logger: Logger,
        access_token: str = None,
        config: Config = None,
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
            logger.info('In HTTPError except\n\n\n\n')
            if http_exception.code == 500 and attempts < HTTP_REQUEST_ATTEMPTS:
                logger.info('500 error returned, retrying request.')
            elif http_exception.code == 500:
                raise UrlAccessFailedWithRetries(url) from http_exception
            else:
                # Not a 500 error, so raise immediately and exit the loop.
                raise UrlAccessFailed(url, http_exception.code) from http_exception

    return response
