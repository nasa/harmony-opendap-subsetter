from logging import getLogger
from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase
from unittest.mock import Mock

from netCDF4 import Dataset
import numpy as np

from varinfo import VarInfoFromDmr, VariableFromDmr

from pymods.spatial import (get_bounding_box_longitudes,
                            get_geographic_index_ranges,
                            get_valid_longitude_range,
                            unwrap_longitude, wrap_longitude)


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

        longitude_variable = Mock(spec=VariableFromDmr)

        for description, valid_min, valid_max, results in test_args:
            with self.subTest(description):
                data = np.ma.masked_array(data=np.linspace(valid_min, valid_max, 361))
                longitude_variable.get_range.return_value = [valid_min, valid_max]

                longitudes = get_bounding_box_longitudes(bounding_box, data,
                                                         longitude_variable)
                self.assertListEqual(longitudes, results)

    def test_wrap_longitude(self):
        """ Ensure that longitudes are correctly mapped to the
            -180 ≤ longitude (degrees) < 180 range.

            `TestCase.assertAlmostEqual` rounds to 7 decimal places.

        """
        longitudes = [['Needs wrapping', 190.0, -170.0],
                      ['Already wrapped', 123.45, 123.45]]

        for description, longitude, expected_longitude in longitudes:
            with self.subTest(description):
                self.assertAlmostEqual(wrap_longitude(longitude),
                                       expected_longitude)

    def test_unwrap_longitudes(self):
        """ Ensure that longitudes are correctly mapped to the
            0 ≤ longitude (degrees) < 360 range.

        """
        longitudes = [['Needs unwrapping', -160.5, 199.5],
                      ['Already unwrapped', 12.34, 12.34]]

        for description, longitude, expected_longitude in longitudes:
            with self.subTest(description):
                self.assertAlmostEqual(unwrap_longitude(longitude),
                                       expected_longitude)

    def test_get_valid_longitude_range(self):
        """ Ensure the valid longitude can be extracted from either the
            valid_range or valid_min and valid_max metadata attributes. Ensure
            that, if these metadata attributes are absent, the longitude range
            can be identified from the data themselves.

        """
        unwrapped_data = np.ma.masked_array(data=np.linspace(0, 360, 361))
        wrapped_data = np.ma.masked_array(data=np.linspace(-180, 180, 361))

        variable_with_range = Mock(spec=VariableFromDmr)
        variable_with_range.get_range.return_value = [-30, 30]

        variable_without_range = Mock(spec=VariableFromDmr)
        variable_without_range.get_range.return_value = None

        with self.subTest('Range data available from VariableFromDmr'):
            valid_range = get_valid_longitude_range(variable_with_range,
                                                    wrapped_data)
            self.assertListEqual(valid_range, [-30, 30])

        with self.subTest('No metadata attributes, data > 180 degrees'):
            valid_range = get_valid_longitude_range(variable_without_range,
                                                    unwrapped_data)
            self.assertListEqual(valid_range, [0, 360])

        with self.subTest('No metadata attributes, data ≤ 180 degrees'):
            valid_range = get_valid_longitude_range(variable_without_range,
                                                    wrapped_data)
            self.assertListEqual(valid_range, [-180, 180])
