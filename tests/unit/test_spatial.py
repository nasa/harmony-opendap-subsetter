from logging import getLogger
from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase

from netCDF4 import Dataset
import numpy as np
from varinfo import VarInfoFromDmr

from pymods.spatial import (get_bounding_box_longitudes,
                            get_geographic_index_ranges, get_longitude_in_grid)


class TestSpatial(TestCase):
    """ A class for testing functions in the pymods.spatial module. """
    @classmethod
    def setUpClass(cls):
        cls.logger = getLogger('tests')
        cls.varinfo = VarInfoFromDmr('tests/data/rssmif16d_example.dmr',
                                     cls.logger,
                                     'tests/data/test_subsetter_config.yml')
        cls.test_dir = 'tests/output'

    def setUp(self):
        self.test_dir = mkdtemp()

    def tearDown(self):
        rmtree(self.test_dir)

    def test_get_geographic_index_ranges(self):
        """ Ensure that correct index ranges can be calculated for:

            - Latitude dimensions
            - Longitude dimensions (continuous ranges)
            - Longitude dimensions (bounding box crossing grid edge)
            - Latitude dimension (descending)
            - Longitude dimension (descending, not crossing grid edge)
            - Values that are exactly halfway between pixels.

            This test will use the valid range of the RSSMIF16D collection,
            such that 0 ≤ longitude (degrees east) ≤ 360.

        """
        test_file_name = f'{self.test_dir}/test.nc'
        bounding_box = [160, 45, 200, 85]
        bounding_box_floats = [160.1, 44.9, 200.1, 84.9]

        with Dataset(test_file_name, 'w', format='NETCDF4') as test_file:
            test_file.createDimension('latitude', size=180)
            test_file.createDimension('longitude', size=360)

            test_file.createVariable('latitude', float,
                                     dimensions=('latitude', ))
            test_file['latitude'][:] = np.linspace(-89.5, 89.5, 180)
            test_file['latitude'].setncatts({'units': 'degrees_north'})

            test_file.createVariable('longitude', float,
                                     dimensions=('longitude', ))
            test_file['longitude'][:] = np.linspace(0.5, 359.5, 360)
            test_file['longitude'].setncatts({'units': 'degrees_east'})

        with self.subTest('Latitude dimension, halfway between pixels'):
            # latitude[134] = 44.5, latitude[135] = 45.5:
            # Southern extent = 45 => index = 135 (min index so round up)
            # latitude[174] = 84.5, latitude[175] = 85.5:
            # Northern extent = 85 => index = 174 (max index so round down)
            self.assertDictEqual(
                get_geographic_index_ranges({'/latitude'}, self.varinfo,
                                            test_file_name, bounding_box),
                {'/latitude': (135, 174)}
            )

        with self.subTest('Latitude dimension, not halfway between pixels'):
            # latitude[134] = 44.5, latitude[135] = 45.5:
            # Southern extent = 44.9 => index = 134
            # latitude[174] = 84.5, latitude[175] = 85.5:
            # Northern extent = 84.9 => index = 174
            self.assertDictEqual(
                get_geographic_index_ranges({'/latitude'}, self.varinfo,
                                            test_file_name,
                                            bounding_box_floats),
                {'/latitude': (134, 174)}
            )

        with self.subTest('Longitude dimension, bounding box within grid'):
            # longitude[159] = 159.5, longitude[160] = 160.5:
            # Western extent = 160 => index = 160 (min index so round up)
            # longitude[199] = 199.5, longitude[200] = 200.5:
            # Eastern extent = 200 => index = 199 (max index so round down)
            self.assertDictEqual(
                get_geographic_index_ranges({'/longitude'}, self.varinfo,
                                            test_file_name, bounding_box),
                {'/longitude': (160, 199)}
            )

        with self.subTest('Longitude, bounding box crosses grid edge'):
            # longitude[339] = 339.5, longitude[340] = 340.5:
            # Western longitude = -20 => 340 => index = 340 (min index, so round up)
            # longitude[19] = 19.5, longitude[20] = 20.5:
            # Eastern longitude = 20 => index 19 (max index, so round down)
            bbox_crossing = [-20, 45, 20, 85]
            self.assertDictEqual(
                get_geographic_index_ranges({'/longitude'}, self.varinfo,
                                            test_file_name, bbox_crossing),
                {'/longitude': (340, 19)}
            )

        with Dataset(test_file_name, 'w', format='NETCDF4') as test_file:
            test_file.createDimension('latitude', size=180)
            test_file.createDimension('longitude', size=360)

            test_file.createVariable('latitude', float,
                                     dimensions=('latitude', ))
            test_file['latitude'][:] = np.linspace(89.5, -89.5, 180)
            test_file['latitude'].setncatts({'units': 'degrees_north'})

            test_file.createVariable('longitude', float,
                                     dimensions=('longitude', ))
            test_file['longitude'][:] = np.linspace(359.5, 0.5, 360)
            test_file['longitude'].setncatts({'units': 'degrees_east'})

        with self.subTest('Descending dimensions, not halfway between pixels'):
            # latitude[4] = 85.5, latitude[5] = 84.5, lat = 84.9 => index = 5
            # latitude[44] = 45.5, latitude[45] = 44.5, lat = 44.9 => index = 45
            # longitude[159] = 200.5, longitude[160] = 199.5, lon = 200.1 => 159
            # longitude[199] = 160.5, longitude[200] = 159.5, lon = 160.1 => 199
            self.assertDictEqual(
                get_geographic_index_ranges({'/latitude', '/longitude'},
                                            self.varinfo, test_file_name,
                                            bounding_box_floats),
                {'/latitude': (5, 45), '/longitude': (159, 199)}
            )

        with self.subTest('Descending dimensions, halfway between pixels'):
            # latitude[4] = 85.5, latitude[5] = 84.5, lat = 85 => index = 5
            # latitude[44] = 45.5, latitude[45] = 44.5, lat = 45 => index = 44
            # longitude[159] = 200.5, longitude[160] = 199.5, lon = 200 => index = 160
            # longitude[199] = 160.5, longitude[200] = 159.5, lon = 160 => index = 199
            self.assertDictEqual(
                get_geographic_index_ranges({'/latitude', '/longitude'},
                                            self.varinfo, test_file_name,
                                            bounding_box),
                {'/latitude': (5, 44), '/longitude': (160, 199)}
            )

    def test_get_bounding_box_longitudes(self):
        """ Ensure the western and eastern extents of a bounding box are
            converted to the correct range according to the range of the
            longitude variable.

            If the variable range is -180 ≤ longitude (degrees) < 180, then the
            bounding box values should remain unconverted. If the variable
            range is 0 ≤ longitude (degrees) < 360, then the bounding box
            values should be converted to this range.

        """
        bounding_box = [-150, -15, -120, 15]

        test_args = [['-180 ≤ lon (deg) < 180', -180, 180, [-150, -120]],
                     ['0 ≤ lon (deg) < 360', 0, 360, [210, 240]]]

        for description, valid_min, valid_max, results in test_args:
            with self.subTest(description):
                data = np.ma.masked_array(data=np.linspace(valid_min, valid_max, 361))
                longitudes = get_bounding_box_longitudes(bounding_box, data)
                self.assertListEqual(longitudes, results)

        partially_wrapped_longitudes = np.linspace(-180, 179.375, 576)

        test_args = [['W = -180, E = -140', -180, -140, [-180, -140]],
                     ['W = 0, E = 179.6875', 0, 179.6875, [0, 179.6875]],
                     ['W = 179.688, E = 180', 179.688, 180, [-180.312, -180]]]

        for description, bbox_west, bbox_east, expected_output in test_args:
            with self.subTest(f'Partial wrapping: {description}'):
                self.assertListEqual(
                    get_bounding_box_longitudes([bbox_west, -15, bbox_east, 15],
                                                partially_wrapped_longitudes),
                    expected_output
                )

    def test_get_longitude_in_grid(self):
        """ Ensure a longitude value is retrieved, where possible, that is
            within the given grid. For example, if longitude = -10 degrees east
            and the grid 0 ≤ longitude (degrees east) ≤ 360, the resulting
            value should be 190 degrees east.

        """
        rss_min, rss_max = (0, 360)
        gpm_min, gpm_max = (-180, 180)
        merra_min, merra_max = (-180.3125, 179.6875)

        test_args = [
            ['RSSMIF16D antimeridian', rss_min, rss_max, -180, 180],
            ['RSSMIF16D negative longitude', rss_min, rss_max, -140, 220],
            ['RSSMIF16D Prime Meridian', rss_min, rss_max, 0, 0],
            ['RSSMIF16D positive longitude', rss_min, rss_max, 40, 40],
            ['RSSMIF16D antimeridian positive', rss_min, rss_max, 180, 180],
            ['GPM antimeridian', gpm_min, gpm_max, -180, -180],
            ['GPM negative longitude', gpm_min, gpm_max, -140, -140],
            ['GPM Prime Meridian', gpm_min, gpm_max, 0, 0],
            ['GPM positive longitude', gpm_min, gpm_max, 40, 40],
            ['GPM antimeridian positive', gpm_min, gpm_max, 180, 180],
            ['MERRA-2 antimeridian', merra_min, merra_max, -180, -180],
            ['MERRA-2 negative longitude', merra_min, merra_max, -140, -140],
            ['MERRA-2 Prime Meridian', merra_min, merra_max, 0, 0],
            ['MERRA-2 positive longitude', merra_min, merra_max, 40, 40],
            ['MERRA-2 antimeridian positive', merra_min, merra_max, 180, -180],
            ['MERRA-2 partial wrapping', merra_min, merra_max, 179.69, -180.31],
            ['MERRA-2 grid_max', merra_min, merra_max, merra_max, merra_max],
            ['Greater than grid max', 0, 10, 12, 12],
            ['Less than grid min', 0, 10, -1, -1],
        ]

        for test, grid_min, grid_max, input_lon, expected_output in test_args:
            with self.subTest(test):
                self.assertEqual(
                    get_longitude_in_grid(grid_min, grid_max, input_lon),
                    expected_output
                )
