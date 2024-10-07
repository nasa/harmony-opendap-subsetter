from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase
from unittest.mock import ANY, call, patch

import numpy as np
from harmony.message import Message
from netCDF4 import Dataset
from numpy.testing import assert_array_equal
from pyproj import CRS
from varinfo import VarInfoFromDmr

from hoss.bbox_utilities import BBox
from hoss.coordinate_utilities import update_dimension_variables
from hoss.spatial import (
    get_bounding_box_longitudes,
    get_geographic_index_range,
    get_longitude_in_grid,
    get_projected_x_y_index_ranges,
    get_spatial_index_ranges,
    get_x_y_index_ranges_from_coordinates,
)


class TestSpatial(TestCase):
    """A class for testing functions in the hoss.spatial module."""

    @classmethod
    def setUpClass(cls):
        cls.varinfo = VarInfoFromDmr(
            'tests/data/rssmif16d_example.dmr',
            config_file='tests/data/test_subsetter_config.json',
        )
        cls.test_dir = 'tests/output'

    def setUp(self):
        self.test_dir = mkdtemp()

    def tearDown(self):
        rmtree(self.test_dir)

    def test_get_spatial_index_ranges_projected(self):
        """Ensure that correct index ranges can be calculated for an ABoVE
        TVPRM granule. This granule has variables that use a grid with an
        Albers Conic Equal Area projection, in the Alaska region.

        """
        harmony_message = Message({'subset': {'bbox': [-160, 68, -145, 70]}})
        above_varinfo = VarInfoFromDmr('tests/data/ABoVE_TVPRM_example.dmr')

        self.assertDictEqual(
            get_spatial_index_ranges(
                {'/NEE', '/x', '/y', '/time'},
                above_varinfo,
                'tests/data/ABoVE_TVPRM_prefetch.nc4',
                harmony_message,
            ),
            {'/x': (37, 56), '/y': (7, 26)},
        )

    def test_get_spatial_index_ranges_geographic(self):
        """Ensure that correct index ranges can be calculated for:

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
        harmony_message_ints = Message({'subset': {'bbox': [160, 45, 200, 85]}})
        harmony_message_floats = Message(
            {'subset': {'bbox': [160.1, 44.9, 200.1, 84.9]}}
        )

        with Dataset(test_file_name, 'w', format='NETCDF4') as test_file:
            test_file.createDimension('latitude', size=180)
            test_file.createDimension('longitude', size=360)

            test_file.createVariable('latitude', float, dimensions=('latitude',))
            test_file['latitude'][:] = np.linspace(-89.5, 89.5, 180)
            test_file['latitude'].setncatts({'units': 'degrees_north'})

            test_file.createVariable('longitude', float, dimensions=('longitude',))
            test_file['longitude'][:] = np.linspace(0.5, 359.5, 360)
            test_file['longitude'].setncatts({'units': 'degrees_east'})

        with self.subTest('Latitude dimension, halfway between pixels'):
            # latitude[134] = 44.5, latitude[135] = 45.5:
            # Southern extent = 45 => index = 135 (min index so round up)
            # latitude[174] = 84.5, latitude[175] = 85.5:
            # Northern extent = 85 => index = 174 (max index so round down)
            self.assertDictEqual(
                get_spatial_index_ranges(
                    {'/latitude'}, self.varinfo, test_file_name, harmony_message_ints
                ),
                {'/latitude': (135, 174)},
            )

        with self.subTest('Latitude dimension, not halfway between pixels'):
            # latitude[134] = 44.5, latitude[135] = 45.5:
            # Southern extent = 44.9 => index = 134
            # latitude[174] = 84.5, latitude[175] = 85.5:
            # Northern extent = 84.9 => index = 174
            self.assertDictEqual(
                get_spatial_index_ranges(
                    {'/latitude'}, self.varinfo, test_file_name, harmony_message_floats
                ),
                {'/latitude': (134, 174)},
            )

        with self.subTest('Longitude dimension, bounding box within grid'):
            # longitude[159] = 159.5, longitude[160] = 160.5:
            # Western extent = 160 => index = 160 (min index so round up)
            # longitude[199] = 199.5, longitude[200] = 200.5:
            # Eastern extent = 200 => index = 199 (max index so round down)
            self.assertDictEqual(
                get_spatial_index_ranges(
                    {'/longitude'}, self.varinfo, test_file_name, harmony_message_ints
                ),
                {'/longitude': (160, 199)},
            )

        with self.subTest('Longitude, bounding box crosses grid edge'):
            # longitude[339] = 339.5, longitude[340] = 340.5:
            # Western longitude = -20 => 340 => index = 340 (min index, so round up)
            # longitude[19] = 19.5, longitude[20] = 20.5:
            # Eastern longitude = 20 => index 19 (max index, so round down)
            harmony_message_crossing = Message({'subset': {'bbox': [-20, 45, 20, 85]}})
            self.assertDictEqual(
                get_spatial_index_ranges(
                    {'/longitude'},
                    self.varinfo,
                    test_file_name,
                    harmony_message_crossing,
                ),
                {'/longitude': (340, 19)},
            )

        with Dataset(test_file_name, 'w', format='NETCDF4') as test_file:
            test_file.createDimension('latitude', size=180)
            test_file.createDimension('longitude', size=360)

            test_file.createVariable('latitude', float, dimensions=('latitude',))
            test_file['latitude'][:] = np.linspace(89.5, -89.5, 180)
            test_file['latitude'].setncatts({'units': 'degrees_north'})

            test_file.createVariable('longitude', float, dimensions=('longitude',))
            test_file['longitude'][:] = np.linspace(359.5, 0.5, 360)
            test_file['longitude'].setncatts({'units': 'degrees_east'})

        with self.subTest('Descending dimensions, not halfway between pixels'):
            # latitude[4] = 85.5, latitude[5] = 84.5, lat = 84.9 => index = 5
            # latitude[44] = 45.5, latitude[45] = 44.5, lat = 44.9 => index = 45
            # longitude[159] = 200.5, longitude[160] = 199.5, lon = 200.1 => 159
            # longitude[199] = 160.5, longitude[200] = 159.5, lon = 160.1 => 199
            self.assertDictEqual(
                get_spatial_index_ranges(
                    {'/latitude', '/longitude'},
                    self.varinfo,
                    test_file_name,
                    harmony_message_floats,
                ),
                {'/latitude': (5, 45), '/longitude': (159, 199)},
            )

        with self.subTest('Descending dimensions, halfway between pixels'):
            # latitude[4] = 85.5, latitude[5] = 84.5, lat = 85 => index = 5
            # latitude[44] = 45.5, latitude[45] = 44.5, lat = 45 => index = 44
            # longitude[159] = 200.5, longitude[160] = 199.5, lon = 200 => index = 160
            # longitude[199] = 160.5, longitude[200] = 159.5, lon = 160 => index = 199
            self.assertDictEqual(
                get_spatial_index_ranges(
                    {'/latitude', '/longitude'},
                    self.varinfo,
                    test_file_name,
                    harmony_message_ints,
                ),
                {'/latitude': (5, 44), '/longitude': (160, 199)},
            )

    @patch('hoss.spatial.get_dimension_index_range')
    @patch('hoss.spatial.get_projected_x_y_extents')
    def test_get_x_y_index_ranges_from_coordinates(
        self,
        mock_get_x_y_extents,
        mock_get_dimension_index_range,
    ):
        """Ensure that x and y index ranges are only requested only when there are
        no projected dimensions and when there are coordinate datasets,
        and the values have not already been calculated.

        The example used in this test is for the SMAP SPL3SMP collection,
        (SMAP L3 Radiometer Global Daily 36 km EASE-Grid Soil Moisture)
        which has a Equal-Area Scalable Earth Grid (EASE-Grid 2.0) CRS for
        a projected grid which is lambert_cylindrical_equal_area projection

        """
        smap_varinfo = VarInfoFromDmr(
            'tests/data/SC_SPL3SMP_009.dmr',
            'SPL3SMP',
            'hoss/hoss_config.json',
        )
        smap_file_path = 'tests/data/SC_SPL3SMP_009_prefetch.nc4'
        expected_index_ranges = {'projected_x': (487, 595), 'projected_y': (9, 38)}
        bbox = BBox(2, 54, 42, 72)
        smap_variable_name = '/Soil_Moisture_Retrieval_Data_AM/surface_flag'

        latitude_coordinate = smap_varinfo.get_variable(
            '/Soil_Moisture_Retrieval_Data_AM/latitude'
        )
        longitude_coordinate = smap_varinfo.get_variable(
            '/Soil_Moisture_Retrieval_Data_AM/longitude'
        )

        crs = CRS.from_cf(
            {
                'false_easting': 0.0,
                'false_northing': 0.0,
                'longitude_of_central_meridian': 0.0,
                'standard_parallel': 30.0,
                'grid_mapping_name': 'lambert_cylindrical_equal_area',
            }
        )

        x_y_extents = {
            'x_min': 192972.56050179302,
            'x_max': 4052423.7705376535,
            'y_min': 5930779.396449475,
            'y_max': 6979878.9118312765,
        }

        mock_get_x_y_extents.return_value = x_y_extents

        # When ranges are derived, they are first calculated for x, then y:
        mock_get_dimension_index_range.side_effect = [(487, 595), (9, 38)]

        with self.subTest(
            'Projected grid from coordinates gets expected dimension ranges'
        ):
            with Dataset(smap_file_path, 'r') as smap_prefetch:
                self.assertDictEqual(
                    get_x_y_index_ranges_from_coordinates(
                        '/Soil_Moisture_Retrieval_Data_AM/surface_flag',
                        smap_varinfo,
                        smap_prefetch,
                        latitude_coordinate,
                        longitude_coordinate,
                        {},
                        bounding_box=bbox,
                        shape_file_path=None,
                    ),
                    expected_index_ranges,
                )

                mock_get_x_y_extents.assert_called_once_with(
                    ANY, ANY, crs, shape_file=None, bounding_box=bbox
                )

                self.assertEqual(mock_get_dimension_index_range.call_count, 2)
                mock_get_dimension_index_range.assert_has_calls(
                    [
                        call(
                            ANY,
                            x_y_extents['x_min'],
                            x_y_extents['x_max'],
                            bounds_values=None,
                        ),
                        call(
                            ANY,
                            x_y_extents['y_min'],
                            x_y_extents['y_max'],
                            bounds_values=None,
                        ),
                    ]
                )

        mock_get_x_y_extents.reset_mock()
        mock_get_dimension_index_range.reset_mock()

        with self.subTest('Function does not rederive known index ranges'):
            with Dataset(smap_file_path, 'r') as smap_prefetch:
                self.assertDictEqual(
                    get_x_y_index_ranges_from_coordinates(
                        '/Soil_Moisture_Retrieval_Data_AM/surface_flag',
                        smap_varinfo,
                        smap_prefetch,
                        latitude_coordinate,
                        longitude_coordinate,
                        expected_index_ranges,
                        bounding_box=bbox,
                        shape_file_path=None,
                    ),
                    {},
                )

            mock_get_x_y_extents.assert_not_called()
            mock_get_dimension_index_range.assert_not_called()

    @patch('hoss.spatial.get_dimension_index_range')
    @patch('hoss.spatial.get_projected_x_y_extents')
    def test_get_projected_x_y_index_ranges(
        self, mock_get_x_y_extents, mock_get_dimension_index_range
    ):
        """Ensure that x and y index ranges are only requested when there are
        projected grid dimensions, and the values have not already been
        calculated.

        The example used in this test is for the ABoVE TVPRM collection,
        which uses an Albers Conical Equal Area CRS for a projected grid,
        with data in Alaska.

        """
        above_varinfo = VarInfoFromDmr('tests/data/ABoVE_TVPRM_example.dmr')
        above_file_path = 'tests/data/ABoVE_TVPRM_prefetch.nc4'
        expected_index_ranges = {'/x': (37, 56), '/y': (7, 26)}
        bbox = BBox(-160, 68, -145, 70)

        crs = CRS.from_cf(
            {
                'false_easting': 0.0,
                'false_northing': 0.0,
                'latitude_of_projection_origin': 40.0,
                'longitude_of_central_meridian': -96.0,
                'standard_parallel': [50.0, 70.0],
                'long_name': 'CRS definition',
                'longitude_of_prime_meridian': 0.0,
                'semi_major_axis': 6378137.0,
                'inverse_flattening': 298.257222101,
                'grid_mapping_name': 'albers_conical_equal_area',
            }
        )

        x_y_extents = {
            'x_min': -2273166.953240025,
            'x_max': -1709569.3224678137,
            'y_min': 3832621.3156695124,
            'y_max': 4425654.159834823,
        }

        mock_get_x_y_extents.return_value = x_y_extents

        # When ranges are derived, they are first calculated for x, then y:
        mock_get_dimension_index_range.side_effect = [(37, 56), (7, 26)]

        with self.subTest('Projected grid gets expected dimension ranges'):
            with Dataset(above_file_path, 'r') as above_prefetch:
                self.assertDictEqual(
                    get_projected_x_y_index_ranges(
                        '/NEE', above_varinfo, above_prefetch, {}, bounding_box=bbox
                    ),
                    expected_index_ranges,
                )

                # Assertions don't like direct comparisons of numpy arrays, so
                # have to extract the call arguments and compare those
                mock_get_x_y_extents.assert_called_once_with(
                    ANY, ANY, crs, shape_file=None, bounding_box=bbox
                )

                actual_x_values = mock_get_x_y_extents.call_args_list[0][0][0]
                actual_y_values = mock_get_x_y_extents.call_args_list[0][0][1]

                assert_array_equal(actual_x_values, above_prefetch['/x'][:])
                assert_array_equal(actual_y_values, above_prefetch['/y'][:])

                self.assertEqual(mock_get_dimension_index_range.call_count, 2)
                mock_get_dimension_index_range.assert_has_calls(
                    [
                        call(
                            ANY,
                            x_y_extents['x_min'],
                            x_y_extents['x_max'],
                            bounds_values=None,
                        ),
                        call(
                            ANY,
                            x_y_extents['y_min'],
                            x_y_extents['y_max'],
                            bounds_values=None,
                        ),
                    ]
                )
                assert_array_equal(
                    mock_get_dimension_index_range.call_args_list[0][0][0],
                    above_prefetch['/x'][:],
                )
                assert_array_equal(
                    mock_get_dimension_index_range.call_args_list[1][0][0],
                    above_prefetch['/y'][:],
                )

        mock_get_x_y_extents.reset_mock()
        mock_get_dimension_index_range.reset_mock()

        with self.subTest('Non projected grid not try to get index ranges'):
            with Dataset(above_file_path, 'r') as above_prefetch:
                self.assertDictEqual(
                    get_projected_x_y_index_ranges(
                        '/x', above_varinfo, above_prefetch, {}, bounding_box=bbox
                    ),
                    {},
                )

            mock_get_x_y_extents.assert_not_called()
            mock_get_dimension_index_range.assert_not_called()

        with self.subTest('Function does not rederive known index ranges'):
            with Dataset(above_file_path, 'r') as above_prefetch:
                self.assertDictEqual(
                    get_projected_x_y_index_ranges(
                        '/NEE',
                        above_varinfo,
                        above_prefetch,
                        expected_index_ranges,
                        bounding_box=bbox,
                    ),
                    {},
                )

            mock_get_x_y_extents.assert_not_called()
            mock_get_dimension_index_range.assert_not_called()

    @patch('hoss.spatial.get_dimension_index_range')
    def test_get_geographic_index_range(self, mock_get_dimension_index_range):
        """Ensure both a latitude and longitude variable is correctly handled.

        The numpy arrays cannot be compared directly as part of the
        `unittest.mock.Mock.assert_called_once_with`, and so require the
        use of `numpy.testing.assert_array_equal`.

        """
        bounding_box = BBox(10, 20, 30, 40)
        mock_get_dimension_index_range.return_value = (1, 2)

        with self.subTest('Latitude variable'):
            with Dataset('tests/data/f16_ssmis_lat_lon.nc', 'r') as prefetch:
                self.assertTupleEqual(
                    get_geographic_index_range(
                        '/latitude', self.varinfo, prefetch, bounding_box
                    ),
                    (1, 2),
                )

                mock_get_dimension_index_range.assert_called_once_with(
                    ANY, bounding_box.south, bounding_box.north, bounds_values=None
                )
                assert_array_equal(
                    mock_get_dimension_index_range.call_args_list[0][0][0],
                    prefetch['/latitude'][:],
                )

            mock_get_dimension_index_range.reset_mock()

        with self.subTest('Longitude variable'):
            with Dataset('tests/data/f16_ssmis_lat_lon.nc', 'r') as prefetch:
                self.assertEqual(
                    get_geographic_index_range(
                        '/longitude', self.varinfo, prefetch, bounding_box
                    ),
                    (1, 2),
                )

                mock_get_dimension_index_range.assert_called_once_with(
                    ANY, bounding_box.west, bounding_box.east, bounds_values=None
                )
                assert_array_equal(
                    mock_get_dimension_index_range.call_args_list[0][0][0],
                    prefetch['/longitude'][:],
                )

            mock_get_dimension_index_range.reset_mock()

    @patch('hoss.spatial.get_dimension_index_range')
    def test_get_geographic_index_range_bounds(self, mock_get_dimension_index_range):
        """Ensure the expected bounds values can be extracted for a variable
        that has the appropriate metadata, and that these bounds values are
        used in the call to `get_dimension_index_range`.

        """
        gpm_varinfo = VarInfoFromDmr(
            'tests/data/GPM_3IMERGHH_example.dmr', short_name='GPM_3IMERGHH'
        )
        bounding_box = BBox(10, 20, 30, 40)
        mock_get_dimension_index_range.return_value = (1, 2)

        with self.subTest('Latitude variable with bounds'):
            with Dataset('tests/data/GPM_3IMERGHH_prefetch.nc4', 'r') as prefetch:
                self.assertEqual(
                    get_geographic_index_range(
                        '/Grid/lat', gpm_varinfo, prefetch, bounding_box
                    ),
                    (1, 2),
                )

                mock_get_dimension_index_range.assert_called_once_with(
                    ANY, bounding_box.south, bounding_box.north, bounds_values=ANY
                )
                assert_array_equal(
                    mock_get_dimension_index_range.call_args_list[0][0][0],
                    prefetch['/Grid/lat'][:],
                )
                assert_array_equal(
                    mock_get_dimension_index_range.call_args_list[0][1][
                        'bounds_values'
                    ],
                    prefetch['/Grid/lat_bnds'][:],
                )

            mock_get_dimension_index_range.reset_mock()

        with self.subTest('Longitude variable with bounds'):
            with Dataset('tests/data/GPM_3IMERGHH_prefetch.nc4', 'r') as prefetch:
                self.assertEqual(
                    get_geographic_index_range(
                        '/Grid/lon', gpm_varinfo, prefetch, bounding_box
                    ),
                    (1, 2),
                )

                mock_get_dimension_index_range.assert_called_once_with(
                    ANY, bounding_box.west, bounding_box.east, bounds_values=ANY
                )
                assert_array_equal(
                    mock_get_dimension_index_range.call_args_list[0][0][0],
                    prefetch['/Grid/lon'][:],
                )
                assert_array_equal(
                    mock_get_dimension_index_range.call_args_list[0][1][
                        'bounds_values'
                    ],
                    prefetch['/Grid/lon_bnds'][:],
                )

            mock_get_dimension_index_range.reset_mock()

    def test_get_bounding_box_longitudes(self):
        """Ensure the western and eastern extents of a bounding box are
        converted to the correct range according to the range of the
        longitude variable.

        If the variable range is -180 ≤ longitude (degrees) < 180, then the
        bounding box values should remain unconverted. If the variable
        range is 0 ≤ longitude (degrees) < 360, then the bounding box
        values should be converted to this range.

        """
        bounding_box = BBox(-150, -15, -120, 15)

        test_args = [
            ['-180 ≤ lon (deg) < 180', -180, 180, [-150, -120]],
            ['0 ≤ lon (deg) < 360', 0, 360, [210, 240]],
        ]

        for description, valid_min, valid_max, results in test_args:
            with self.subTest(description):
                data = np.ma.masked_array(data=np.linspace(valid_min, valid_max, 361))
                longitudes = get_bounding_box_longitudes(bounding_box, data)
                self.assertListEqual(longitudes, results)

        partially_wrapped_longitudes = np.linspace(-180, 179.375, 576)

        test_args = [
            ['W = -180, E = -140', -180, -140, [-180, -140]],
            ['W = 0, E = 179.6875', 0, 179.6875, [0, 179.6875]],
            ['W = 179.688, E = 180', 179.688, 180, [-180.312, -180]],
        ]

        for description, bbox_west, bbox_east, expected_output in test_args:
            with self.subTest(f'Partial wrapping: {description}'):
                input_bounding_box = BBox(bbox_west, -15, bbox_east, 15)
                self.assertListEqual(
                    get_bounding_box_longitudes(
                        input_bounding_box, partially_wrapped_longitudes
                    ),
                    expected_output,
                )

    def test_get_longitude_in_grid(self):
        """Ensure a longitude value is retrieved, where possible, that is
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
                    expected_output,
                )
