"""Utility classes used to extend the unittest capabilities"""

from collections import namedtuple
from datetime import datetime
from typing import List
from unittest.mock import MagicMock

from harmony_service_lib.util import bbox_to_geometry
from pystac import Asset, Catalog, Item

Granule = namedtuple('Granule', ['url', 'media_type', 'roles'])


def write_dmr(output_dir: str, content: str):
    """A helper function to write out the content of a `.dmr`, when the
    `harmony.util.download` function is called. This will be called as
    a side-effect to the mock for that function.

    """
    dmr_name = f'{output_dir}/downloaded.dmr'

    with open(dmr_name, 'w', encoding='utf-8') as file_handler:
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


def create_stac(granules: List[Granule]) -> Catalog:
    """Create a SpatioTemporal Asset Catalog (STAC). These are used as inputs
    for Harmony requests, containing the URL and other information for
    input granules.

    For simplicity the geometry and temporal properties of each item are
    set to default values, as only the URL, media type and role are used by
    HOSS.

    """
    catalog = Catalog(id='input', description='test input')

    for granule_index, granule in enumerate(granules):
        item = Item(
            id=f'granule_{granule_index}',
            geometry=bbox_to_geometry([-180, -90, 180, 90]),
            bbox=[-180, -90, 180, 90],
            datetime=datetime(2020, 1, 1),
            properties=None,
        )
        item.add_asset(
            'input_data',
            Asset(granule.url, media_type=granule.media_type, roles=granule.roles),
        )
        catalog.add_item(item)

    return catalog
