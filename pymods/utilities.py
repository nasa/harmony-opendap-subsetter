""" This module contains lower-level functionality that can be abstracted into
    functions. Primarily this improves readability of the source code, and
    allows finer-grained unit testing of each smaller part of functionality.

"""
from typing import Optional, Tuple
import mimetypes

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
