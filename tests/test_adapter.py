"""These end-to-end tests simulate a full invocation of the Harmony
service, instantiating a HossAdapter using a message and STAC.
Each test mocks calls to OPeNDAP, and has an extensive set of
assertions to ensure the output is as expected, and the expected
requests were made to OPeNDAP.

"""

from shutil import copy, rmtree
from tempfile import mkdtemp
from typing import Dict, Set
from unittest import TestCase
from unittest.mock import ANY, Mock, call, patch

from harmony_service_lib.exceptions import NoDataException, NoRetryException
from harmony_service_lib.message import Message
from harmony_service_lib.util import HarmonyException, config
from netCDF4 import Dataset
from numpy.testing import assert_array_equal
from pystac import Catalog
from varinfo import VarInfoFromDmr

from hoss.adapter import HossAdapter
from hoss.exceptions import InvalidVariableRequest
from tests.utilities import Granule, create_stac, write_dmr


class TestHossEndToEnd(TestCase):

    @classmethod
    def setUpClass(cls):
        """Test fixture that can be set once for all tests in the class."""
        cls.granule_url = 'https://harmony.uat.earthdata.nasa.gov/opendap_url'
        cls.input_stac = create_stac(
            [Granule(cls.granule_url, None, ['opendap', 'data'])]
        )
        cls.atl03_variable = '/gt1r/geophys_corr/geoid'
        cls.gpm_variable = '/Grid/precipitationCal'
        cls.rssmif16d_variable = '/wind_speed'
        cls.atl16_variable = '/global_asr_obs_grid'
        cls.staging_location = 's3://example-bucket/'

        with open('tests/data/ATL03_example.dmr', 'r') as file_handler:
            cls.atl03_dmr = file_handler.read()

        with open('tests/data/rssmif16d_example.dmr', 'r') as file_handler:
            cls.rssmif16d_dmr = file_handler.read()

        with open('tests/data/M2T1NXSLV_example.dmr', 'r') as file_handler:
            cls.m2t1nxslv_dmr = file_handler.read()

        with open('tests/data/GPM_3IMERGHH_example.dmr', 'r') as file_handler:
            cls.gpm_imerghh_dmr = file_handler.read()

        with open('tests/data/ATL16_prefetch.dmr', 'r') as file_handler:
            cls.atl16_dmr = file_handler.read()

    def setUp(self):
        """Have to mock mkdtemp, to know where to put mock .dmr content."""
        self.tmp_dir = mkdtemp()
        self.config = config(validate=False)

    def tearDown(self):
        rmtree(self.tmp_dir)

    def assert_valid_request_data(
        self, request_data: Dict, expected_variables: Set[str]
    ):
        """Check the contents of the request data sent to the OPeNDAP server
        when retrieving a NetCDF-4 file. This should ensure that a URL
        encoded constraint expression was sent, and that all the expected
        variables (potentially with index ranges) were included.

        This custom class method is used because the constraint expressions
        are constructed from sets. The order of variables in the set, and
        therefore the constraint expression string, cannot be guaranteed.

        """
        opendap_separator = '%3B'
        self.assertIn('dap4.ce', request_data)
        requested_variables = set(request_data['dap4.ce'].split(opendap_separator))
        self.assertSetEqual(requested_variables, expected_variables)

    def assert_expected_output_catalog(
        self,
        catalog: Catalog,
        expected_href: str,
        expected_title: str,
        expected_mimetype='application/x-netcdf4',
    ):
        """Check the contents of the Harmony output STAC. It should have a
        single data item, containing an asset with the supplied URL and
        title.

        """
        items = list(catalog.get_items())

        self.assertEqual(len(items), 1)
        self.assertListEqual(list(items[0].assets.keys()), ['data'])

        actual_catalog = items[0].assets['data'].to_dict()
        expected_catalog = {
            'href': expected_href,
            'title': expected_title,
            'type': expected_mimetype,
            'roles': ['data'],
        }

        # Check all the dictionary values match except for the href value.
        self.assertTrue(
            all(
                actual_catalog[key] == expected_catalog[key]
                for key in actual_catalog
                if key != 'href'
            )
        )

        # The href value must be compared separately because it can contain
        # a constraint expression in the case where an unexecuted OPeNDAP URL
        # is requested, where the variable order is not consistent.
        self.assertEqual(
            sorted(actual_catalog['href']), sorted(expected_catalog['href'])
        )

    @patch('hoss.utilities.uuid4')
    @patch('hoss.adapter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('hoss.utilities.util_download')
    @patch('hoss.adapter.stage')
    def test_opendap_url_end_to_end(
        self, mock_stage, mock_util_download, mock_rmtree, mock_mkdtemp, mock_uuid4
    ):
        """Ensure HOSS will run an unexecuted OPeNDAP URL request end-to-end,
        only mocking the HTTP responses, and the output interactions
        with Harmony.

        """
        expected_title = 'OPeNAP Request URL'
        expected_dap4 = '.dap.nc4?dap4.ce=%2Flongitude%5B60%3A119%5D%3B%2Fwind_speed%5B%5D%5B540%3A599%5D%5B60%3A119%5D%3B%2Flatitude%5B540%3A599%5D%3B%2Ftime'
        expected_opendap_url = f'{self.granule_url}{expected_dap4}'

        mock_uuid4.return_value = Mock(hex='uuid')
        mock_mkdtemp.return_value = self.tmp_dir

        mimetype = 'application/x-netcdf4;profile=opendap_url'

        dmr_path = write_dmr(self.tmp_dir, self.rssmif16d_dmr)

        dimensions_path = f'{self.tmp_dir}/dimensions.nc4'
        copy('tests/data/f16_ssmis_lat_lon.nc', dimensions_path)

        all_variables_path = f'{self.tmp_dir}/variables.nc4'
        copy('tests/data/f16_ssmis_geo.nc', all_variables_path)

        mock_util_download.side_effect = [dmr_path, dimensions_path, all_variables_path]

        message = Message(
            {
                'accessToken': 'fake-token',
                'callback': 'https://example.com/',
                'sources': [
                    {
                        'collection': 'C1234567890-EEDTEST',
                        'shortName': 'RSSMIF16D',
                        'variables': [
                            {
                                'id': '',
                                'name': self.rssmif16d_variable,
                                'fullPath': self.rssmif16d_variable,
                            }
                        ],
                    }
                ],
                'stagingLocation': self.staging_location,
                'subset': {'bbox': [15, 45, 30, 60]},
                'format': {'mime': mimetype},
                'user': 'auser',
            }
        )

        hoss = HossAdapter(message, config=config(False), catalog=self.input_stac)

        _, output_catalog = hoss.invoke()

        # Ensure that there is a single item in the output catalog with the
        # expected asset:
        self.assert_expected_output_catalog(
            output_catalog, expected_opendap_url, expected_title, mimetype
        )
        mock_stage.assert_not_called()
        mock_rmtree.assert_called_once_with(self.tmp_dir)

    @patch('hoss.utilities.uuid4')
    @patch('hoss.adapter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('hoss.utilities.util_download')
    @patch('hoss.adapter.stage')
    def test_non_spatial_end_to_end(
        self, mock_stage, mock_util_download, mock_rmtree, mock_mkdtemp, mock_uuid
    ):
        """Ensure HOSS will run end-to-end, only mocking the HTTP responses,
        and the output interactions with Harmony.

        This test should only perform a variable subset.

        """
        expected_output_basename = 'opendap_url_gt1r_geophys_corr_geoid_subsetted.nc4'
        expected_staged_url = f'{self.staging_location}{expected_output_basename}'

        mock_uuid.return_value = Mock(hex='uuid')
        mock_mkdtemp.return_value = self.tmp_dir
        mock_stage.return_value = expected_staged_url

        dmr_path = write_dmr(self.tmp_dir, self.atl03_dmr)

        downloaded_nc4_path = f'{self.tmp_dir}/opendap_url_subset.nc4'
        # There needs to be a physical file present to be renamed by Harmony.
        # The contents are not accessed.
        copy('tests/data/ATL03_example.dmr', downloaded_nc4_path)

        mock_util_download.side_effect = [dmr_path, downloaded_nc4_path]

        message = Message(
            {
                'accessToken': 'fake-token',
                'callback': 'https://example.com/',
                'sources': [
                    {
                        'collection': 'C1234567890-EEDTEST',
                        'shortName': 'ATL03',
                        'variables': [
                            {
                                'id': '',
                                'name': self.atl03_variable,
                                'fullPath': self.atl03_variable,
                            }
                        ],
                    }
                ],
                'stagingLocation': self.staging_location,
                'user': 'fhaise',
            }
        )

        hoss = HossAdapter(message, config=config(False), catalog=self.input_stac)

        _, output_catalog = hoss.invoke()

        # Ensure that there is a single item in the output catalog with the
        # expected asset:
        self.assert_expected_output_catalog(
            output_catalog, expected_staged_url, expected_output_basename
        )

        # Ensure the correct number of downloads were requested from OPeNDAP:
        # the first should be the `.dmr`. The second should be the required
        # variables. (There is no dimension prefetch in this type of request)
        # The requested variables for the final output are from a set, and so
        # their order cannot be guaranteed. Instead, `data` is matched to
        # `ANY`, and the constraint expression is tested separately.
        self.assertEqual(mock_util_download.call_count, 2)
        mock_util_download.assert_has_calls(
            [
                call(
                    f'{self.granule_url}.dmr.xml',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=None,
                    cfg=hoss.config,
                ),
                call(
                    f'{self.granule_url}.dap.nc4',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=ANY,
                    cfg=hoss.config,
                ),
            ]
        )

        # Ensure the constraint expression contains all the required variables.
        post_data = mock_util_download.call_args_list[1][1].get('data', {})
        self.assert_valid_request_data(
            post_data,
            {
                '%2Fgt1r%2Fgeolocation%2Fdelta_time',
                '%2Fgt1r%2Fgeolocation%2Freference_photon_lon',
                '%2Fgt1r%2Fgeolocation%2Fpodppd_flag',
                '%2Fgt1r%2Fgeophys_corr%2Fdelta_time',
                '%2Fgt1r%2Fgeolocation%2Freference_photon_lat',
                '%2Fgt1r%2Fgeophys_corr%2Fgeoid',
            },
        )

        # Ensure the output was staged with the expected file name
        mock_stage.assert_called_once_with(
            f'{self.tmp_dir}/uuid.nc4',
            expected_output_basename,
            'application/x-netcdf4',
            location=self.staging_location,
            logger=hoss.logger,
        )
        mock_rmtree.assert_called_once_with(self.tmp_dir)

    @patch('hoss.dimension_utilities.get_fill_slice')
    @patch('hoss.utilities.uuid4')
    @patch('hoss.adapter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('hoss.utilities.util_download')
    @patch('hoss.adapter.stage')
    def test_geo_bbox_end_to_end(
        self,
        mock_stage,
        mock_util_download,
        mock_rmtree,
        mock_mkdtemp,
        mock_uuid,
        mock_get_fill_slice,
    ):
        """Ensure a request with a bounding box will be correctly processed
        for a geographically gridded collection, requesting only the
        expected variables, with index ranges corresponding to the bounding
        box specified.

        """
        expected_output_basename = 'opendap_url_wind_speed_subsetted.nc4'
        expected_staged_url = f'{self.staging_location}{expected_output_basename}'

        mock_uuid.side_effect = [Mock(hex='uuid'), Mock(hex='uuid2')]
        mock_mkdtemp.return_value = self.tmp_dir
        mock_stage.return_value = expected_staged_url

        dmr_path = write_dmr(self.tmp_dir, self.rssmif16d_dmr)

        dimensions_path = f'{self.tmp_dir}/dimensions.nc4'
        copy('tests/data/f16_ssmis_lat_lon.nc', dimensions_path)

        all_variables_path = f'{self.tmp_dir}/variables.nc4'
        copy('tests/data/f16_ssmis_geo.nc', all_variables_path)

        mock_util_download.side_effect = [dmr_path, dimensions_path, all_variables_path]

        message = Message(
            {
                'accessToken': 'fake-token',
                'callback': 'https://example.com/',
                'sources': [
                    {
                        'collection': 'C1234567890-EEDTEST',
                        'shortName': 'RSSMIF16D',
                        'variables': [
                            {
                                'id': '',
                                'name': self.rssmif16d_variable,
                                'fullPath': self.rssmif16d_variable,
                            }
                        ],
                    }
                ],
                'stagingLocation': self.staging_location,
                'subset': {'bbox': [-30, 45, -15, 60]},
                'user': 'jlovell',
            }
        )

        hoss = HossAdapter(message, config=config(False), catalog=self.input_stac)

        _, output_catalog = hoss.invoke()

        # Ensure that there is a single item in the output catalog with the
        # expected asset:
        self.assert_expected_output_catalog(
            output_catalog, expected_staged_url, expected_output_basename
        )

        # Ensure the expected requests were made against OPeNDAP.
        # The first should be the `.dmr`. The second should fetch a NetCDF-4
        # file containing the full 1-D dimension variables only, and the third
        # should retrieve the final NetCDF-4 with all required variables.
        # The requested variables for the dimension prefetch and the final
        # output are from a set, and so their order cannot be guaranteed.
        # Instead, `data` is matched to `ANY`, and the constraint expression is
        # tested separately.
        self.assertEqual(mock_util_download.call_count, 3)
        mock_util_download.assert_has_calls(
            [
                call(
                    f'{self.granule_url}.dmr.xml',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=None,
                    cfg=hoss.config,
                ),
                call(
                    f'{self.granule_url}.dap.nc4',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=ANY,
                    cfg=hoss.config,
                ),
                call(
                    f'{self.granule_url}.dap.nc4',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=ANY,
                    cfg=hoss.config,
                ),
            ]
        )

        # Ensure the constraint expression for dimensions data included only
        # geographic or temporal variables with no index ranges
        dimensions_data = mock_util_download.call_args_list[1][1].get('data', {})
        self.assert_valid_request_data(
            dimensions_data, {'%2Flatitude', '%2Flongitude', '%2Ftime'}
        )
        # Ensure the constraint expression contains all the required variables.
        # /wind_speed[][540:599][1320:1379], /time, /longitude[1320:1379]
        # /latitude[540:599]
        index_range_data = mock_util_download.call_args_list[2][1].get('data', {})
        self.assert_valid_request_data(
            index_range_data,
            {
                '%2Ftime',
                '%2Flatitude%5B540%3A599%5D',
                '%2Flongitude%5B1320%3A1379%5D',
                '%2Fwind_speed%5B%5D%5B540%3A599%5D%5B1320%3A1379%5D',
            },
        )

        # Ensure the output was staged with the expected file name
        mock_stage.assert_called_once_with(
            f'{self.tmp_dir}/uuid2.nc4',
            expected_output_basename,
            'application/x-netcdf4',
            location=self.staging_location,
            logger=hoss.logger,
        )
        mock_rmtree.assert_called_once_with(self.tmp_dir)

        # Ensure no variables were filled
        mock_get_fill_slice.assert_not_called()

    @patch('hoss.dimension_utilities.get_fill_slice')
    @patch('hoss.utilities.uuid4')
    @patch('hoss.adapter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('hoss.utilities.util_download')
    @patch('hoss.adapter.stage')
    def test_bbox_geo_descending_latitude(
        self,
        mock_stage,
        mock_util_download,
        mock_rmtree,
        mock_mkdtemp,
        mock_uuid,
        mock_get_fill_slice,
    ):
        """Ensure a request with a bounding box will be correctly processed,
        for a geographically gridded collection, requesting only the
        expected variables, with index ranges corresponding to the bounding
        box specified. The latitude dimension returned from the geographic
        dimensions request to OPeNDAP will be descending. This test is to
        ensure the correct dimension indices are identified and the correct
        DAP4 constraint expression is built.

        """
        expected_output_basename = 'opendap_url_wind_speed_subsetted.nc4'
        expected_staged_url = f'{self.staging_location}{expected_output_basename}'

        mock_uuid.side_effect = [Mock(hex='uuid'), Mock(hex='uuid2')]
        mock_mkdtemp.return_value = self.tmp_dir
        mock_stage.return_value = expected_staged_url

        dmr_path = write_dmr(self.tmp_dir, self.rssmif16d_dmr)

        dimensions_path = f'{self.tmp_dir}/dimensions.nc4'
        copy('tests/data/f16_ssmis_lat_lon_desc.nc', dimensions_path)

        all_variables_path = f'{self.tmp_dir}/variables.nc4'
        copy('tests/data/f16_ssmis_geo_desc.nc', all_variables_path)

        mock_util_download.side_effect = [dmr_path, dimensions_path, all_variables_path]

        message = Message(
            {
                'accessToken': 'fake-token',
                'callback': 'https://example.com/',
                'sources': [
                    {
                        'collection': 'C1234567890-EEDTEST',
                        'shortName': 'RSSMIF16D',
                        'variables': [
                            {
                                'id': '',
                                'name': self.rssmif16d_variable,
                                'fullPath': self.rssmif16d_variable,
                            }
                        ],
                    }
                ],
                'stagingLocation': self.staging_location,
                'subset': {'bbox': [-30, 45, -15, 60]},
                'user': 'cduke',
            }
        )

        hoss = HossAdapter(message, config=config(False), catalog=self.input_stac)
        _, output_catalog = hoss.invoke()

        # Ensure that there is a single item in the output catalog with the
        # expected asset:
        self.assert_expected_output_catalog(
            output_catalog, expected_staged_url, expected_output_basename
        )

        # Ensure the expected requests were made against OPeNDAP.
        # See related comment in self.test_geo_bbox_end_to_end
        self.assertEqual(mock_util_download.call_count, 3)
        mock_util_download.assert_has_calls(
            [
                call(
                    f'{self.granule_url}.dmr.xml',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=None,
                    cfg=hoss.config,
                ),
                call(
                    f'{self.granule_url}.dap.nc4',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=ANY,
                    cfg=hoss.config,
                ),
                call(
                    f'{self.granule_url}.dap.nc4',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=ANY,
                    cfg=hoss.config,
                ),
            ]
        )

        # Ensure the constraint expression for dimensions data included only
        # geographic or temporal variables with no index ranges
        dimensions_data = mock_util_download.call_args_list[1][1].get('data', {})
        self.assert_valid_request_data(
            dimensions_data, {'%2Flatitude', '%2Flongitude', '%2Ftime'}
        )
        # Ensure the constraint expression contains all the required variables.
        # /wind_speed[][120:179][1320:1379], /time, /longitude[1320:1379]
        # /latitude[120:179]
        index_range_data = mock_util_download.call_args_list[2][1].get('data', {})
        self.assert_valid_request_data(
            index_range_data,
            {
                '%2Ftime',
                '%2Flatitude%5B120%3A179%5D',
                '%2Flongitude%5B1320%3A1379%5D',
                '%2Fwind_speed%5B%5D%5B120%3A179%5D%5B1320%3A1379%5D',
            },
        )

        # Ensure the output was staged with the expected file name
        mock_stage.assert_called_once_with(
            f'{self.tmp_dir}/uuid2.nc4',
            expected_output_basename,
            'application/x-netcdf4',
            location=self.staging_location,
            logger=hoss.logger,
        )
        mock_rmtree.assert_called_once_with(self.tmp_dir)

        # Ensure no variables were filled:
        mock_get_fill_slice.assert_not_called()

    @patch('hoss.utilities.uuid4')
    @patch('hoss.adapter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('hoss.utilities.util_download')
    @patch('hoss.adapter.stage')
    def test_geo_bbox_crossing_grid_edge(
        self, mock_stage, mock_util_download, mock_rmtree, mock_mkdtemp, mock_uuid
    ):
        """Ensure a request with a bounding box that crosses a longitude edge
        (360 degrees east) requests the expected variables from OPeNDAP and
        does so only in the expected latitude range. The full longitude
        range should be requested for all variables, with filling applied
        outside of the bounding box region.

        """
        expected_output_basename = 'opendap_url_wind_speed_subsetted.nc4'
        expected_staged_url = f'{self.staging_location}{expected_output_basename}'

        mock_uuid.side_effect = [Mock(hex='uuid'), Mock(hex='uuid2')]
        mock_mkdtemp.return_value = self.tmp_dir
        mock_stage.return_value = expected_staged_url

        dmr_path = write_dmr(self.tmp_dir, self.rssmif16d_dmr)

        dimensions_path = f'{self.tmp_dir}/dimensions.nc4'
        copy('tests/data/f16_ssmis_lat_lon.nc', dimensions_path)

        unfilled_path = f'{self.tmp_dir}/variables.nc4'
        copy('tests/data/f16_ssmis_unfilled.nc', unfilled_path)

        mock_util_download.side_effect = [dmr_path, dimensions_path, unfilled_path]

        message = Message(
            {
                'accessToken': 'fake-token',
                'callback': 'https://example.com/',
                'sources': [
                    {
                        'collection': 'C1234567890-EEDTEST',
                        'shortName': 'RSSMIF16D',
                        'variables': [
                            {
                                'id': '',
                                'name': self.rssmif16d_variable,
                                'fullPath': self.rssmif16d_variable,
                            }
                        ],
                    }
                ],
                'stagingLocation': self.staging_location,
                'subset': {'bbox': [-7.5, -60, 7.5, -45]},
                'user': 'jswiggert',
            }
        )

        hoss = HossAdapter(message, config=config(False), catalog=self.input_stac)
        _, output_catalog = hoss.invoke()

        # Ensure that there is a single item in the output catalog with the
        # expected asset:
        self.assert_expected_output_catalog(
            output_catalog, expected_staged_url, expected_output_basename
        )

        # Ensure the expected requests were made against OPeNDAP.
        # See related comment in self.test_geo_bbox_end_to_end
        self.assertEqual(mock_util_download.call_count, 3)
        mock_util_download.assert_has_calls(
            [
                call(
                    f'{self.granule_url}.dmr.xml',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=None,
                    cfg=hoss.config,
                ),
                call(
                    f'{self.granule_url}.dap.nc4',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=ANY,
                    cfg=hoss.config,
                ),
                call(
                    f'{self.granule_url}.dap.nc4',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=ANY,
                    cfg=hoss.config,
                ),
            ]
        )

        # Ensure the constraint expression for dimensions data included only
        # geographic or temporal variables with no index ranges
        dimensions_data = mock_util_download.call_args_list[1][1].get('data', {})
        self.assert_valid_request_data(
            dimensions_data, {'%2Flatitude', '%2Flongitude', '%2Ftime'}
        )
        # Ensure the constraint expression contains all the required variables.
        # /wind_speed[][120:179][], /time, /longitude (full range),
        # /latitude[120:179]
        index_range_data = mock_util_download.call_args_list[2][1].get('data', {})
        self.assert_valid_request_data(
            index_range_data,
            {
                '%2Ftime',
                '%2Flatitude%5B120%3A179%5D',
                '%2Flongitude',
                '%2Fwind_speed%5B%5D%5B120%3A179%5D%5B%5D',
            },
        )

        # Ensure the output was staged with the expected file name
        mock_stage.assert_called_once_with(
            f'{self.tmp_dir}/uuid2.nc4',
            expected_output_basename,
            'application/x-netcdf4',
            location=self.staging_location,
            logger=hoss.logger,
        )
        mock_rmtree.assert_called_once_with(self.tmp_dir)

        # Ensure the final output was correctly filled (the unfilled file is
        # filled in place):
        expected_output = Dataset('tests/data/f16_ssmis_filled.nc', 'r')
        actual_output = Dataset(f'{self.tmp_dir}/uuid2.nc4', 'r')

        for variable_name, expected_variable in expected_output.variables.items():
            self.assertIn(variable_name, actual_output.variables)
            assert_array_equal(actual_output[variable_name][:], expected_variable[:])

        expected_output.close()
        actual_output.close()

    @patch('hoss.dimension_utilities.get_fill_slice')
    @patch('hoss.utilities.uuid4')
    @patch('hoss.adapter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('hoss.utilities.util_download')
    @patch('hoss.adapter.stage')
    def test_geo_bbox(
        self,
        mock_stage,
        mock_util_download,
        mock_rmtree,
        mock_mkdtemp,
        mock_uuid,
        mock_get_fill_slice,
    ):
        """Ensure requests with particular bounding box edge-cases return the
        correct pixel ranges:

        * Single point, N=S, W=E, inside a pixel, retrieves that single
          pixel.
        * Single point, N=S, W=E, in corner of 4 pixels retrieves all 4
          surrounding pixels.
        * Line, N=S, W < E, where the latitude is inside a pixel, retrieves
          a single row of pixels.
        * Line, N > S, W=E, where longitude is between pixels, retrieves
          two columns of pixels, corresponding to those which touch the
          line.

        """
        point_in_pixel = [-29.99, 45.01, -29.99, 45.01]
        point_between_pixels = [-30, 45, -30, 45]
        line_in_pixels = [-30, -14.99, -15, -14.99]
        line_between_pixels = [-30, 45, -30, 60]

        range_point_in_pixel = {
            '%2Ftime',
            '%2Flatitude%5B540%3A540%5D',
            '%2Flongitude%5B1320%3A1320%5D',
            '%2Fwind_speed%5B%5D%5B540%3A540%5D%5B1320%3A1320%5D',
        }

        range_point_between_pixels = {
            '%2Ftime',
            '%2Flatitude%5B539%3A540%5D',
            '%2Flongitude%5B1319%3A1320%5D',
            '%2Fwind_speed%5B%5D%5B539%3A540%5D%5B1319%3A1320%5D',
        }

        range_line_in_pixels = {
            '%2Ftime',
            '%2Flatitude%5B300%3A300%5D',
            '%2Flongitude%5B1320%3A1379%5D',
            '%2Fwind_speed%5B%5D%5B300%3A300%5D%5B1320%3A1379%5D',
        }

        range_line_between_pixels = {
            '%2Ftime',
            '%2Flatitude%5B540%3A599%5D',
            '%2Flongitude%5B1319%3A1320%5D',
            '%2Fwind_speed%5B%5D%5B540%3A599%5D%5B1319%3A1320%5D',
        }

        test_args = [
            ['Point is inside single pixel', point_in_pixel, range_point_in_pixel],
            [
                'Point in corner of 4 pixels',
                point_between_pixels,
                range_point_between_pixels,
            ],
            ['Line through single row', line_in_pixels, range_line_in_pixels],
            [
                'Line between two columns',
                line_between_pixels,
                range_line_between_pixels,
            ],
        ]

        for description, bounding_box, expected_index_ranges in test_args:
            with self.subTest(description):
                expected_output_basename = 'opendap_url_wind_speed_subsetted.nc4'
                expected_staged_url = (
                    f'{self.staging_location}{expected_output_basename}'
                )
                mock_uuid.side_effect = [Mock(hex='uuid'), Mock(hex='uuid2')]
                mock_mkdtemp.return_value = self.tmp_dir
                mock_stage.return_value = expected_staged_url

                dmr_path = write_dmr(self.tmp_dir, self.rssmif16d_dmr)

                dimensions_path = f'{self.tmp_dir}/dimensions.nc4'
                copy('tests/data/f16_ssmis_lat_lon.nc', dimensions_path)

                all_variables_path = f'{self.tmp_dir}/variables.nc4'
                copy('tests/data/f16_ssmis_geo.nc', all_variables_path)

                mock_util_download.side_effect = [
                    dmr_path,
                    dimensions_path,
                    all_variables_path,
                ]

                message = Message(
                    {
                        'accessToken': 'fake-token',
                        'callback': 'https://example.com/',
                        'sources': [
                            {
                                'collection': 'C1234567890-EEDTEST',
                                'shortName': 'RSSMIF16D',
                                'variables': [
                                    {
                                        'id': '',
                                        'name': self.rssmif16d_variable,
                                        'fullPath': self.rssmif16d_variable,
                                    }
                                ],
                            }
                        ],
                        'stagingLocation': self.staging_location,
                        'subset': {'bbox': bounding_box},
                        'user': 'jaaron',
                    }
                )

                hoss = HossAdapter(
                    message, config=config(False), catalog=self.input_stac
                )
                _, output_catalog = hoss.invoke()

                # Ensure that there is a single item in the output catalog with
                # the expected asset:
                self.assert_expected_output_catalog(
                    output_catalog, expected_staged_url, expected_output_basename
                )

                # Ensure the expected requests were made against OPeNDAP.
                # See related comment in self.test_geo_bbox_end_to_end
                self.assertEqual(mock_util_download.call_count, 3)
                mock_util_download.assert_has_calls(
                    [
                        call(
                            f'{self.granule_url}.dmr.xml',
                            self.tmp_dir,
                            hoss.logger,
                            access_token=message.accessToken,
                            data=None,
                            cfg=hoss.config,
                        ),
                        call(
                            f'{self.granule_url}.dap.nc4',
                            self.tmp_dir,
                            hoss.logger,
                            access_token=message.accessToken,
                            data=ANY,
                            cfg=hoss.config,
                        ),
                        call(
                            f'{self.granule_url}.dap.nc4',
                            self.tmp_dir,
                            hoss.logger,
                            access_token=message.accessToken,
                            data=ANY,
                            cfg=hoss.config,
                        ),
                    ]
                )

                # Ensure the constraint expression for dimensions data included
                # only geographic or temporal variables with no index ranges
                dimensions_data = mock_util_download.call_args_list[1][1].get(
                    'data', {}
                )
                self.assert_valid_request_data(
                    dimensions_data, {'%2Flatitude', '%2Flongitude', '%2Ftime'}
                )
                # Ensure the constraint expression contains all the required variables.
                index_range_data = mock_util_download.call_args_list[2][1].get(
                    'data', {}
                )
                self.assert_valid_request_data(index_range_data, expected_index_ranges)

                # Ensure the output was staged with the expected file name
                mock_stage.assert_called_once_with(
                    f'{self.tmp_dir}/uuid2.nc4',
                    expected_output_basename,
                    'application/x-netcdf4',
                    location=self.staging_location,
                    logger=hoss.logger,
                )
                mock_rmtree.assert_called_once_with(self.tmp_dir)

                # Ensure no variables were filled
                mock_get_fill_slice.assert_not_called()

            mock_mkdtemp.reset_mock()
            mock_uuid.reset_mock()
            mock_util_download.reset_mock()
            mock_stage.reset_mock()
            mock_get_fill_slice.reset_mock()
            mock_rmtree.reset_mock()

    @patch('hoss.dimension_utilities.get_fill_slice')
    @patch('hoss.utilities.uuid4')
    @patch('hoss.adapter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('hoss.utilities.util_download')
    @patch('hoss.adapter.stage')
    def test_spatial_bbox_no_variables(
        self,
        mock_stage,
        mock_util_download,
        mock_rmtree,
        mock_mkdtemp,
        mock_uuid,
        mock_get_fill_slice,
    ):
        """Ensure a request with a bounding box that does not specify any
        variables will retrieve all variables, but limited to the range
        specified by the bounding box.

        """
        expected_output_basename = 'opendap_url_subsetted.nc4'
        expected_staged_url = f'{self.staging_location}{expected_output_basename}'

        mock_uuid.side_effect = [Mock(hex='uuid'), Mock(hex='uuid2')]
        mock_mkdtemp.return_value = self.tmp_dir
        mock_stage.return_value = expected_staged_url

        dmr_path = write_dmr(self.tmp_dir, self.rssmif16d_dmr)

        dimensions_path = f'{self.tmp_dir}/dimensions.nc4'
        copy('tests/data/f16_ssmis_lat_lon.nc', dimensions_path)

        all_variables_path = f'{self.tmp_dir}/variables.nc4'
        copy('tests/data/f16_ssmis_geo_no_vars.nc', all_variables_path)

        mock_util_download.side_effect = [dmr_path, dimensions_path, all_variables_path]

        message = Message(
            {
                'accessToken': 'fake-token',
                'callback': 'https://example.com/',
                'sources': [
                    {'collection': 'C1234567890-EEDTEST', 'shortName': 'RSSMIF16D'}
                ],
                'stagingLocation': self.staging_location,
                'subset': {'bbox': [-30, 45, -15, 60]},
                'user': 'kerwinj',
            }
        )

        hoss = HossAdapter(message, config=config(False), catalog=self.input_stac)
        _, output_catalog = hoss.invoke()

        # Ensure that there is a single item in the output catalog with the
        # expected asset:
        self.assert_expected_output_catalog(
            output_catalog, expected_staged_url, expected_output_basename
        )

        # Ensure the expected requests were made against OPeNDAP.
        # See related comment in self.test_geo_bbox_end_to_end
        self.assertEqual(mock_util_download.call_count, 3)
        mock_util_download.assert_has_calls(
            [
                call(
                    f'{self.granule_url}.dmr.xml',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=None,
                    cfg=hoss.config,
                ),
                call(
                    f'{self.granule_url}.dap.nc4',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=ANY,
                    cfg=hoss.config,
                ),
                call(
                    f'{self.granule_url}.dap.nc4',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=ANY,
                    cfg=hoss.config,
                ),
            ]
        )

        # Ensure the constraint expression for dimensions data included only
        # geographic or temporal variables with no index ranges
        dimensions_data = mock_util_download.call_args_list[1][1].get('data', {})
        self.assert_valid_request_data(
            dimensions_data, {'%2Flatitude', '%2Flongitude', '%2Ftime'}
        )
        # Ensure the constraint expression contains all variables.
        # /atmosphere_cloud_liquid_water_content[][540:599][1320:1379],
        # /atmosphere_water_vapor_content[][540:599][1320:1379],
        # /rainfall_rate[][540:599][1320:1379],
        # /sst_dtime[][540:599][1320:1379], /wind_speed[][540:599][1320:1379],
        # /time, /longitude[1320:1379], /latitude[540:599]
        index_range_data = mock_util_download.call_args_list[2][1].get('data', {})
        self.assert_valid_request_data(
            index_range_data,
            {
                '%2Ftime',
                '%2Flatitude%5B540%3A599%5D',
                '%2Flongitude%5B1320%3A1379%5D',
                '%2Fatmosphere_cloud_liquid_water_content%5B%5D%5B540%3A599%5D%5B1320%3A1379%5D',
                '%2Fatmosphere_water_vapor_content%5B%5D%5B540%3A599%5D%5B1320%3A1379%5D',
                '%2Frainfall_rate%5B%5D%5B540%3A599%5D%5B1320%3A1379%5D',
                '%2Fsst_dtime%5B%5D%5B540%3A599%5D%5B1320%3A1379%5D',
                '%2Fwind_speed%5B%5D%5B540%3A599%5D%5B1320%3A1379%5D',
            },
        )

        # Ensure the output was staged with the expected file name
        mock_stage.assert_called_once_with(
            f'{self.tmp_dir}/uuid2.nc4',
            expected_output_basename,
            'application/x-netcdf4',
            location=self.staging_location,
            logger=hoss.logger,
        )
        mock_rmtree.assert_called_once_with(self.tmp_dir)

        # Ensure no variables were filled:
        mock_get_fill_slice.assert_not_called()

    @patch('hoss.dimension_utilities.get_fill_slice')
    @patch('hoss.utilities.uuid4')
    @patch('hoss.adapter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('hoss.utilities.util_download')
    @patch('hoss.adapter.stage')
    def test_temporal_end_to_end(
        self,
        mock_stage,
        mock_util_download,
        mock_rmtree,
        mock_mkdtemp,
        mock_uuid,
        mock_get_fill_slice,
    ):
        """Ensure a request with a temporal range will retrieve variables,
        but limited to the range specified by the temporal range.

        The example granule has 24 hourly time slices, starting with
        2021-01-10T00:30:00.

        """
        expected_output_basename = 'opendap_url_PS_subsetted.nc4'
        expected_staged_url = f'{self.staging_location}{expected_output_basename}'

        mock_uuid.side_effect = [Mock(hex='uuid'), Mock(hex='uuid2')]
        mock_mkdtemp.return_value = self.tmp_dir
        mock_stage.return_value = expected_staged_url

        dmr_path = write_dmr(self.tmp_dir, self.m2t1nxslv_dmr)

        dimensions_path = f'{self.tmp_dir}/dimensions.nc4'
        copy('tests/data/M2T1NXSLV_prefetch.nc4', dimensions_path)

        temporal_variables_path = f'{self.tmp_dir}/temporal_variables.nc4'
        copy('tests/data/M2T1NXSLV_temporal.nc4', temporal_variables_path)

        mock_util_download.side_effect = [
            dmr_path,
            dimensions_path,
            temporal_variables_path,
        ]

        message = Message(
            {
                'accessToken': 'fake-token',
                'callback': 'https://example.com/',
                'sources': [
                    {
                        'collection': 'C1234567890-EEDTEST',
                        'shortName': 'M2T1NXSLV',
                        'variables': [{'id': '', 'name': '/PS', 'fullPath': '/PS'}],
                    }
                ],
                'stagingLocation': self.staging_location,
                'temporal': {
                    'start': '2021-01-10T01:00:00',
                    'end': '2021-01-10T03:00:00',
                },
                'user': 'jyoung',
            }
        )

        hoss = HossAdapter(message, config=config(False), catalog=self.input_stac)

        _, output_catalog = hoss.invoke()

        # Ensure that there is a single item in the output catalog with the
        # expected asset:
        self.assert_expected_output_catalog(
            output_catalog, expected_staged_url, expected_output_basename
        )

        # Ensure the expected requests were made against OPeNDAP.
        # See related comment in self.test_geo_bbox_end_to_end
        self.assertEqual(mock_util_download.call_count, 3)
        mock_util_download.assert_has_calls(
            [
                call(
                    f'{self.granule_url}.dmr.xml',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=None,
                    cfg=hoss.config,
                ),
                call(
                    f'{self.granule_url}.dap.nc4',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=ANY,
                    cfg=hoss.config,
                ),
                call(
                    f'{self.granule_url}.dap.nc4',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=ANY,
                    cfg=hoss.config,
                ),
            ]
        )

        # Ensure the constraint expression for dimensions data included only
        # geographic or temporal variables with no index ranges
        dimensions_data = mock_util_download.call_args_list[1][1].get('data', {})
        self.assert_valid_request_data(dimensions_data, {'%2Flat', '%2Flon', '%2Ftime'})
        # Ensure the constraint expression contains all the required variables.
        # /PS[1:2][][], /time[1:2], /lon, /lat
        index_range_data = mock_util_download.call_args_list[2][1].get('data', {})
        self.assert_valid_request_data(
            index_range_data,
            {'%2Ftime%5B1%3A2%5D', '%2Flat', '%2Flon', '%2FPS%5B1%3A2%5D%5B%5D%5B%5D'},
        )

        # Ensure the output was staged with the expected file name
        mock_stage.assert_called_once_with(
            f'{self.tmp_dir}/uuid2.nc4',
            expected_output_basename,
            'application/x-netcdf4',
            location=self.staging_location,
            logger=hoss.logger,
        )
        mock_rmtree.assert_called_once_with(self.tmp_dir)

        # Ensure no variables were filled
        mock_get_fill_slice.assert_not_called()

    @patch('hoss.dimension_utilities.get_fill_slice')
    @patch('hoss.utilities.uuid4')
    @patch('hoss.adapter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('hoss.utilities.util_download')
    @patch('hoss.adapter.stage')
    def test_temporal_all_variables(
        self,
        mock_stage,
        mock_util_download,
        mock_rmtree,
        mock_mkdtemp,
        mock_uuid,
        mock_get_fill_slice,
    ):
        """Ensure a request with a temporal range and no specified variables
        will retrieve the expected output. Note - because a temporal range
        is specified, HOSS will need to perform an index range subset. This
        means that the prefetch will still have to occur, and all variables
        with the temporal grid dimension will need to include their index
        ranges in the final DAP4 constraint expression.

        The example granule has 24 hourly time slices, starting with
        2021-01-10T00:30:00.

        """
        expected_output_basename = 'opendap_url_subsetted.nc4'
        expected_staged_url = f'{self.staging_location}{expected_output_basename}'

        mock_uuid.side_effect = [Mock(hex='uuid'), Mock(hex='uuid2')]
        mock_mkdtemp.return_value = self.tmp_dir
        mock_stage.return_value = expected_staged_url

        dmr_path = write_dmr(self.tmp_dir, self.m2t1nxslv_dmr)

        dimensions_path = f'{self.tmp_dir}/dimensions.nc4'
        copy('tests/data/M2T1NXSLV_prefetch.nc4', dimensions_path)

        temporal_variables_path = f'{self.tmp_dir}/temporal_variables.nc4'
        copy('tests/data/M2T1NXSLV_temporal.nc4', temporal_variables_path)

        mock_util_download.side_effect = [
            dmr_path,
            dimensions_path,
            temporal_variables_path,
        ]

        message = Message(
            {
                'accessToken': 'fake-token',
                'callback': 'https://example.com/',
                'sources': [
                    {'collection': 'C1234567890-EEDTEST', 'shortName': 'M2T1NXSLV'}
                ],
                'stagingLocation': self.staging_location,
                'subset': None,
                'temporal': {
                    'start': '2021-01-10T01:00:00',
                    'end': '2021-01-10T03:00:00',
                },
                'user': 'jyoung',
            }
        )

        hoss = HossAdapter(message, config=config(False), catalog=self.input_stac)
        _, output_catalog = hoss.invoke()

        # Ensure that there is a single item in the output catalog with the
        # expected asset:
        self.assert_expected_output_catalog(
            output_catalog, expected_staged_url, expected_output_basename
        )

        # Ensure the expected requests were made against OPeNDAP.
        # See related comment in self.test_geo_bbox_end_to_end
        self.assertEqual(mock_util_download.call_count, 3)
        mock_util_download.assert_has_calls(
            [
                call(
                    f'{self.granule_url}.dmr.xml',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=None,
                    cfg=hoss.config,
                ),
                call(
                    f'{self.granule_url}.dap.nc4',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=ANY,
                    cfg=hoss.config,
                ),
                call(
                    f'{self.granule_url}.dap.nc4',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=ANY,
                    cfg=hoss.config,
                ),
            ]
        )

        # Ensure the constraint expression for dimensions data included only
        # geographic or temporal variables with no index ranges
        dimensions_data = mock_util_download.call_args_list[1][1].get('data', {})
        self.assert_valid_request_data(dimensions_data, {'%2Flat', '%2Flon', '%2Ftime'})
        # Ensure the constraint expression contains all the required variables.
        # /<science_variable>[1:2][][], /time[1:2], /lon, /lat
        index_range_data = mock_util_download.call_args_list[2][1].get('data', {})
        self.assert_valid_request_data(
            index_range_data,
            {
                '%2Ftime%5B1%3A2%5D',
                '%2Flat',
                '%2Flon',
                '%2FCLDPRS%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FCLDTMP%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FDISPH%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FH1000%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FH250%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FH500%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FH850%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FPBLTOP%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FPS%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FOMEGA500%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FQ250%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FQ500%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FQ850%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FQV10M%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FQV2M%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FSLP%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FT10M%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FT250%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FT2M%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FT2MDEW%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FT2MWET%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FT500%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FT850%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FTO3%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FTOX%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FTQL%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FTQI%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FTQV%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FTROPPB%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FTROPPV%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FTROPQ%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FTROPT%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FTROPPT%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FTS%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FU10M%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FU250%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FU2M%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FU500%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FU50M%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FU850%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FV10M%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FV250%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FV2M%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FV500%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FV50M%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FV850%5B1%3A2%5D%5B%5D%5B%5D',
                '%2FZLCL%5B1%3A2%5D%5B%5D%5B%5D',
            },
        )

        # Ensure the output was staged with the expected file name
        mock_stage.assert_called_once_with(
            f'{self.tmp_dir}/uuid2.nc4',
            expected_output_basename,
            'application/x-netcdf4',
            location=self.staging_location,
            logger=hoss.logger,
        )
        mock_rmtree.assert_called_once_with(self.tmp_dir)

        # Ensure no variables were filled
        mock_get_fill_slice.assert_not_called()

    @patch('hoss.dimension_utilities.get_fill_slice')
    @patch('hoss.utilities.uuid4')
    @patch('hoss.adapter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('hoss.utilities.util_download')
    @patch('hoss.adapter.stage')
    def test_bbox_temporal_end_to_end(
        self,
        mock_stage,
        mock_util_download,
        mock_rmtree,
        mock_mkdtemp,
        mock_uuid,
        mock_get_fill_slice,
    ):
        """Ensure a request with both a bounding box and a temporal range will
        retrieve variables, but limited to the ranges specified by the
        bounding box and the temporal range.

        """
        expected_output_basename = 'opendap_url_PS_subsetted.nc4'
        expected_staged_url = f'{self.staging_location}{expected_output_basename}'

        mock_uuid.side_effect = [Mock(hex='uuid'), Mock(hex='uuid2')]
        mock_mkdtemp.return_value = self.tmp_dir
        mock_stage.return_value = expected_staged_url

        dmr_path = write_dmr(self.tmp_dir, self.m2t1nxslv_dmr)

        dimensions_path = f'{self.tmp_dir}/dimensions.nc4'
        copy('tests/data/M2T1NXSLV_prefetch.nc4', dimensions_path)

        geo_temporal_path = f'{self.tmp_dir}/geo_temporal.nc4'
        copy('tests/data/M2T1NXSLV_temporal.nc4', geo_temporal_path)

        mock_util_download.side_effect = [dmr_path, dimensions_path, geo_temporal_path]

        message = Message(
            {
                'accessToken': 'fake-token',
                'callback': 'https://example.com/',
                'sources': [
                    {
                        'collection': 'C1234567890-EEDTEST',
                        'shortName': 'M2T1NXSLV',
                        'variables': [{'id': '', 'name': '/PS', 'fullPath': '/PS'}],
                    }
                ],
                'stagingLocation': self.staging_location,
                'subset': {'bbox': [40, -30, 50, -20]},
                'temporal': {
                    'start': '2021-01-10T01:00:00',
                    'end': '2021-01-10T03:00:00',
                },
                'user': 'jyoung',
            }
        )

        hoss = HossAdapter(message, config=config(False), catalog=self.input_stac)
        _, output_catalog = hoss.invoke()

        # Ensure that there is a single item in the output catalog with the
        # expected asset:
        self.assert_expected_output_catalog(
            output_catalog, expected_staged_url, expected_output_basename
        )

        # Ensure the expected requests were made against OPeNDAP.
        # See related comment in self.test_geo_bbox_end_to_end
        self.assertEqual(mock_util_download.call_count, 3)
        mock_util_download.assert_has_calls(
            [
                call(
                    f'{self.granule_url}.dmr.xml',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=None,
                    cfg=hoss.config,
                ),
                call(
                    f'{self.granule_url}.dap.nc4',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=ANY,
                    cfg=hoss.config,
                ),
                call(
                    f'{self.granule_url}.dap.nc4',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=ANY,
                    cfg=hoss.config,
                ),
            ]
        )

        # Ensure the constraint expression for dimensions data included only
        # geographic or temporal variables with no index ranges
        dimensions_data = mock_util_download.call_args_list[1][1].get('data', {})
        self.assert_valid_request_data(dimensions_data, {'%2Flat', '%2Flon', '%2Ftime'})
        # Ensure the constraint expression contains all the required variables.
        # /PS[1:2][120:140][352:368], /time[1:2], /lon[352:368], /lat[120:140]
        index_range_data = mock_util_download.call_args_list[2][1].get('data', {})
        self.assert_valid_request_data(
            index_range_data,
            {
                '%2Ftime%5B1%3A2%5D',
                '%2Flat%5B120%3A140%5D',
                '%2Flon%5B352%3A368%5D',
                '%2FPS%5B1%3A2%5D%5B120%3A140%5D%5B352%3A368%5D',
            },
        )

        # Ensure the output was staged with the expected file name
        mock_stage.assert_called_once_with(
            f'{self.tmp_dir}/uuid2.nc4',
            expected_output_basename,
            'application/x-netcdf4',
            location=self.staging_location,
            logger=hoss.logger,
        )
        mock_rmtree.assert_called_once_with(self.tmp_dir)

        # Ensure no variables were filled
        mock_get_fill_slice.assert_not_called()

    @patch('hoss.dimension_utilities.get_fill_slice')
    @patch('hoss.utilities.uuid4')
    @patch('hoss.adapter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('hoss.bbox_utilities.download')
    @patch('hoss.utilities.util_download')
    @patch('hoss.adapter.stage')
    def test_geo_shapefile_end_to_end(
        self,
        mock_stage,
        mock_util_download,
        mock_geojson_download,
        mock_rmtree,
        mock_mkdtemp,
        mock_uuid,
        mock_get_fill_slice,
    ):
        """Ensure a request with a shape file specified against a
        geographically gridded collection will retrieve variables, but
        limited to the ranges of a bounding box that encloses the specified
        GeoJSON shape.

        """
        expected_output_basename = 'opendap_url_wind_speed_subsetted.nc4'
        expected_staged_url = f'{self.staging_location}{expected_output_basename}'

        mock_uuid.side_effect = [Mock(hex='uuid'), Mock(hex='uuid2')]
        mock_mkdtemp.return_value = self.tmp_dir
        mock_stage.return_value = expected_staged_url

        dmr_path = write_dmr(self.tmp_dir, self.rssmif16d_dmr)

        dimensions_path = f'{self.tmp_dir}/dimensions.nc4'
        copy('tests/data/f16_ssmis_lat_lon.nc', dimensions_path)

        all_variables_path = f'{self.tmp_dir}/variables.nc4'
        copy('tests/data/f16_ssmis_geo.nc', all_variables_path)

        geojson_path = f'{self.tmp_dir}/polygon.geo.json'
        copy('tests/geojson_examples/polygon.geo.json', geojson_path)

        shape_file_url = 'www.example.com/polygon.geo.json'
        mock_geojson_download.return_value = geojson_path
        mock_util_download.side_effect = [dmr_path, dimensions_path, all_variables_path]

        message = Message(
            {
                'accessToken': 'fake-token',
                'callback': 'https://example.com/',
                'sources': [
                    {
                        'collection': 'C1234567890-EEDTEST',
                        'shortName': 'RSSMIF16D',
                        'variables': [
                            {
                                'id': '',
                                'name': self.rssmif16d_variable,
                                'fullPath': self.rssmif16d_variable,
                            }
                        ],
                    }
                ],
                'stagingLocation': self.staging_location,
                'subset': {
                    'shape': {'href': shape_file_url, 'type': 'application/geo+json'}
                },
                'user': 'dscott',
            }
        )

        hoss = HossAdapter(message, config=config(False), catalog=self.input_stac)
        _, output_catalog = hoss.invoke()

        # Ensure that there is a single item in the output catalog with the
        # expected asset:
        self.assert_expected_output_catalog(
            output_catalog, expected_staged_url, expected_output_basename
        )

        # Ensure the shape file in the Harmony message was downloaded:
        mock_geojson_download.assert_called_once_with(
            shape_file_url,
            self.tmp_dir,
            logger=hoss.logger,
            access_token=message.accessToken,
            cfg=hoss.config,
        )

        # Ensure the expected requests were made against OPeNDAP.
        # See related comment in self.test_geo_bbox_end_to_end
        self.assertEqual(mock_util_download.call_count, 3)
        mock_util_download.assert_has_calls(
            [
                call(
                    f'{self.granule_url}.dmr.xml',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=None,
                    cfg=hoss.config,
                ),
                call(
                    f'{self.granule_url}.dap.nc4',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=ANY,
                    cfg=hoss.config,
                ),
                call(
                    f'{self.granule_url}.dap.nc4',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=ANY,
                    cfg=hoss.config,
                ),
            ]
        )

        # Ensure the constraint expression for dimensions data included only
        # geographic or temporal variables with no index ranges
        dimensions_data = mock_util_download.call_args_list[1][1].get('data', {})
        self.assert_valid_request_data(
            dimensions_data, {'%2Flatitude', '%2Flongitude', '%2Ftime'}
        )

        # Ensure the constraint expression contains all the required variables;
        # the polygon in the GeoJSON is the state of Utah:
        # /wind_speed[][508:527][983:1003], /time, /longitude[983:1003]
        # /latitude[508:527]
        index_range_data = mock_util_download.call_args_list[2][1].get('data', {})
        self.assert_valid_request_data(
            index_range_data,
            {
                '%2Ftime',
                '%2Flatitude%5B508%3A527%5D',
                '%2Flongitude%5B983%3A1003%5D',
                '%2Fwind_speed%5B%5D%5B508%3A527%5D%5B983%3A1003%5D',
            },
        )

        # Ensure the output was staged with the expected file name
        mock_stage.assert_called_once_with(
            f'{self.tmp_dir}/uuid2.nc4',
            expected_output_basename,
            'application/x-netcdf4',
            location=self.staging_location,
            logger=hoss.logger,
        )
        mock_rmtree.assert_called_once_with(self.tmp_dir)

        # Ensure no variables were filled
        mock_get_fill_slice.assert_not_called()

    @patch('hoss.dimension_utilities.get_fill_slice')
    @patch('hoss.utilities.uuid4')
    @patch('hoss.adapter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('hoss.bbox_utilities.download')
    @patch('hoss.utilities.util_download')
    @patch('hoss.adapter.stage')
    def test_geo_shapefile_all_variables(
        self,
        mock_stage,
        mock_util_download,
        mock_geojson_download,
        mock_rmtree,
        mock_mkdtemp,
        mock_uuid,
        mock_get_fill_slice,
    ):
        """Ensure an all variable request with a shape file specified will
        retrieve all variables, but limited to the ranges of a bounding box
        that encloses the specified GeoJSON shape. This request uses a
        collection that is geographically gridded.

        Because a shape file is specified, index range subsetting will be
        performed, so a prefetch request will be performed, and the final
        DAP4 constraint expression will include all variables with index
        ranges.

        """
        expected_output_basename = 'opendap_url_subsetted.nc4'
        expected_staged_url = f'{self.staging_location}{expected_output_basename}'

        mock_uuid.side_effect = [Mock(hex='uuid'), Mock(hex='uuid2')]
        mock_mkdtemp.return_value = self.tmp_dir
        mock_stage.return_value = expected_staged_url

        dmr_path = write_dmr(self.tmp_dir, self.rssmif16d_dmr)

        dimensions_path = f'{self.tmp_dir}/dimensions.nc4'
        copy('tests/data/f16_ssmis_lat_lon.nc', dimensions_path)

        all_variables_path = f'{self.tmp_dir}/variables.nc4'
        copy('tests/data/f16_ssmis_geo.nc', all_variables_path)

        geojson_path = f'{self.tmp_dir}/polygon.geo.json'
        copy('tests/geojson_examples/polygon.geo.json', geojson_path)

        shape_file_url = 'www.example.com/polygon.geo.json'
        mock_geojson_download.return_value = geojson_path
        mock_util_download.side_effect = [dmr_path, dimensions_path, all_variables_path]

        message = Message(
            {
                'accessToken': 'fake-token',
                'callback': 'https://example.com/',
                'sources': [
                    {'collection': 'C1234567890-EEDTEST', 'shortName': 'RSSMIF16D'}
                ],
                'stagingLocation': self.staging_location,
                'subset': {
                    'shape': {'href': shape_file_url, 'type': 'application/geo+json'}
                },
                'user': 'dscott',
            }
        )

        hoss = HossAdapter(message, config=config(False), catalog=self.input_stac)
        _, output_catalog = hoss.invoke()

        # Ensure that there is a single item in the output catalog with the
        # expected asset:
        self.assert_expected_output_catalog(
            output_catalog, expected_staged_url, expected_output_basename
        )

        # Ensure the shape file in the Harmony message was downloaded:
        mock_geojson_download.assert_called_once_with(
            shape_file_url,
            self.tmp_dir,
            logger=hoss.logger,
            access_token=message.accessToken,
            cfg=hoss.config,
        )

        # Ensure the expected requests were made against OPeNDAP.
        # See related comment in self.test_geo_bbox_end_to_end
        self.assertEqual(mock_util_download.call_count, 3)
        mock_util_download.assert_has_calls(
            [
                call(
                    f'{self.granule_url}.dmr.xml',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=None,
                    cfg=hoss.config,
                ),
                call(
                    f'{self.granule_url}.dap.nc4',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=ANY,
                    cfg=hoss.config,
                ),
                call(
                    f'{self.granule_url}.dap.nc4',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=ANY,
                    cfg=hoss.config,
                ),
            ]
        )

        # Ensure the constraint expression for dimensions data included only
        # geographic or temporal variables with no index ranges
        dimensions_data = mock_util_download.call_args_list[1][1].get('data', {})
        self.assert_valid_request_data(
            dimensions_data, {'%2Flatitude', '%2Flongitude', '%2Ftime'}
        )

        # Ensure the constraint expression contains all the required variables;
        # the polygon in the GeoJSON is the state of Utah:
        # /<science_variable>[][508:527][983:1003], /time, /longitude[983:1003]
        # /latitude[508:527]
        index_range_data = mock_util_download.call_args_list[2][1].get('data', {})
        self.assert_valid_request_data(
            index_range_data,
            {
                '%2Ftime',
                '%2Flatitude%5B508%3A527%5D',
                '%2Flongitude%5B983%3A1003%5D',
                '%2Fatmosphere_cloud_liquid_water_content%5B%5D%5B508%3A527%5D%5B983%3A1003%5D',
                '%2Fatmosphere_water_vapor_content%5B%5D%5B508%3A527%5D%5B983%3A1003%5D',
                '%2Frainfall_rate%5B%5D%5B508%3A527%5D%5B983%3A1003%5D',
                '%2Fsst_dtime%5B%5D%5B508%3A527%5D%5B983%3A1003%5D',
                '%2Fwind_speed%5B%5D%5B508%3A527%5D%5B983%3A1003%5D',
            },
        )

        # Ensure the output was staged with the expected file name
        mock_stage.assert_called_once_with(
            f'{self.tmp_dir}/uuid2.nc4',
            expected_output_basename,
            'application/x-netcdf4',
            location=self.staging_location,
            logger=hoss.logger,
        )
        mock_rmtree.assert_called_once_with(self.tmp_dir)

        # Ensure no variables were filled
        mock_get_fill_slice.assert_not_called()

    @patch('hoss.dimension_utilities.get_fill_slice')
    @patch('hoss.utilities.uuid4')
    @patch('hoss.adapter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('hoss.bbox_utilities.download')
    @patch('hoss.utilities.util_download')
    @patch('hoss.adapter.stage')
    def test_bbox_precedence_end_to_end(
        self,
        mock_stage,
        mock_util_download,
        mock_geojson_download,
        mock_rmtree,
        mock_mkdtemp,
        mock_uuid,
        mock_get_fill_slice,
    ):
        """Ensure a request with a bounding box will be correctly processed,
        requesting only the expected variables, with index ranges
        corresponding to the bounding box specified.

        """
        expected_output_basename = 'opendap_url_wind_speed_subsetted.nc4'
        expected_staged_url = f'{self.staging_location}{expected_output_basename}'

        mock_uuid.side_effect = [Mock(hex='uuid'), Mock(hex='uuid2')]
        mock_mkdtemp.return_value = self.tmp_dir
        mock_stage.return_value = expected_staged_url

        dmr_path = write_dmr(self.tmp_dir, self.rssmif16d_dmr)

        dimensions_path = f'{self.tmp_dir}/dimensions.nc4'
        copy('tests/data/f16_ssmis_lat_lon.nc', dimensions_path)

        all_variables_path = f'{self.tmp_dir}/variables.nc4'
        copy('tests/data/f16_ssmis_geo.nc', all_variables_path)

        geojson_path = f'{self.tmp_dir}/polygon.geo.json'
        copy('tests/geojson_examples/polygon.geo.json', geojson_path)

        shape_file_url = 'www.example.com/polygon.geo.json'
        mock_geojson_download.return_value = geojson_path
        mock_util_download.side_effect = [dmr_path, dimensions_path, all_variables_path]

        message = Message(
            {
                'accessToken': 'fake-token',
                'callback': 'https://example.com/',
                'sources': [
                    {
                        'collection': 'C1234567890-EEDTEST',
                        'shortName': 'RSSMIF16D',
                        'variables': [
                            {
                                'id': '',
                                'name': self.rssmif16d_variable,
                                'fullPath': self.rssmif16d_variable,
                            }
                        ],
                    }
                ],
                'stagingLocation': self.staging_location,
                'subset': {
                    'bbox': [-30, 45, -15, 60],
                    'shape': {'href': shape_file_url, 'type': 'application/geo+json'},
                },
                'user': 'aworden',
            }
        )

        hoss = HossAdapter(message, config=config(False), catalog=self.input_stac)
        _, output_catalog = hoss.invoke()

        # Ensure that there is a single item in the output catalog with the
        # expected asset:
        self.assert_expected_output_catalog(
            output_catalog, expected_staged_url, expected_output_basename
        )

        # Ensure the shape file in the Harmony message was downloaded (the
        # logic giving the bounding box precedence over the shape file occurs
        # in `hoss/subset.py`, after the shape file has already been
        # downloaded - however, that file will not be used.
        mock_geojson_download.assert_called_once_with(
            shape_file_url,
            self.tmp_dir,
            logger=hoss.logger,
            access_token=message.accessToken,
            cfg=hoss.config,
        )

        # Ensure the expected requests were made against OPeNDAP.
        # See related comment in self.test_geo_bbox_end_to_end
        self.assertEqual(mock_util_download.call_count, 3)
        mock_util_download.assert_has_calls(
            [
                call(
                    f'{self.granule_url}.dmr.xml',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=None,
                    cfg=hoss.config,
                ),
                call(
                    f'{self.granule_url}.dap.nc4',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=ANY,
                    cfg=hoss.config,
                ),
                call(
                    f'{self.granule_url}.dap.nc4',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=ANY,
                    cfg=hoss.config,
                ),
            ]
        )

        # Ensure the constraint expression for dimensions data included only
        # geographic or temporal variables with no index ranges
        dimensions_data = mock_util_download.call_args_list[1][1].get('data', {})
        self.assert_valid_request_data(
            dimensions_data, {'%2Flatitude', '%2Flongitude', '%2Ftime'}
        )
        # Ensure the constraint expression contains all the required variables.
        # The index ranges correspond to the bounding box specified in the
        # Harmony message, not the GeoJSON shape file.
        # /wind_speed[][540:599][1320:1379], /time, /longitude[1320:1379]
        # /latitude[540:599]
        index_range_data = mock_util_download.call_args_list[2][1].get('data', {})
        self.assert_valid_request_data(
            index_range_data,
            {
                '%2Ftime',
                '%2Flatitude%5B540%3A599%5D',
                '%2Flongitude%5B1320%3A1379%5D',
                '%2Fwind_speed%5B%5D%5B540%3A599%5D%5B1320%3A1379%5D',
            },
        )

        # Ensure the output was staged with the expected file name
        mock_stage.assert_called_once_with(
            f'{self.tmp_dir}/uuid2.nc4',
            expected_output_basename,
            'application/x-netcdf4',
            location=self.staging_location,
            logger=hoss.logger,
        )
        mock_rmtree.assert_called_once_with(self.tmp_dir)

        # Ensure no variables were filled
        mock_get_fill_slice.assert_not_called()

    @patch('hoss.dimension_utilities.get_fill_slice')
    @patch('hoss.utilities.uuid4')
    @patch('hoss.adapter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('hoss.utilities.util_download')
    @patch('hoss.adapter.stage')
    def test_geo_dimensions(
        self,
        mock_stage,
        mock_util_download,
        mock_rmtree,
        mock_mkdtemp,
        mock_uuid,
        mock_get_fill_slice,
    ):
        """Ensure a request with explicitly specified dimension extents will
        be correctly processed, requesting only the expected variables,
        with index ranges corresponding to the extents specified.

        To minimise test data in the repository, this test uses geographic
        dimension of latitude and longitude, but within the
        `subset.dimensions` region of the inbound Harmony message.

        """
        expected_output_basename = 'opendap_url_wind_speed_subsetted.nc4'
        expected_staged_url = f'{self.staging_location}{expected_output_basename}'

        mock_uuid.side_effect = [Mock(hex='uuid'), Mock(hex='uuid2')]
        mock_mkdtemp.return_value = self.tmp_dir
        mock_stage.return_value = expected_staged_url

        dmr_path = write_dmr(self.tmp_dir, self.rssmif16d_dmr)

        dimensions_path = f'{self.tmp_dir}/dimensions.nc4'
        copy('tests/data/f16_ssmis_lat_lon.nc', dimensions_path)

        all_variables_path = f'{self.tmp_dir}/variables.nc4'
        copy('tests/data/f16_ssmis_geo.nc', all_variables_path)

        mock_util_download.side_effect = [dmr_path, dimensions_path, all_variables_path]

        message = Message(
            {
                'accessToken': 'fake-token',
                'callback': 'https://example.com/',
                'sources': [
                    {
                        'collection': 'C1234567890-EEDTEST',
                        'shortName': 'RSSMIF16D',
                        'variables': [
                            {
                                'id': '',
                                'name': self.rssmif16d_variable,
                                'fullPath': self.rssmif16d_variable,
                            }
                        ],
                    }
                ],
                'stagingLocation': self.staging_location,
                'subset': {
                    'dimensions': [
                        {'name': 'latitude', 'min': 45, 'max': 60},
                        {'name': 'longitude', 'min': 15, 'max': 30},
                    ]
                },
                'user': 'blightyear',
            }
        )

        hoss = HossAdapter(message, config=config(False), catalog=self.input_stac)
        _, output_catalog = hoss.invoke()

        # Ensure that there is a single item in the output catalog with the
        # expected asset:
        self.assert_expected_output_catalog(
            output_catalog, expected_staged_url, expected_output_basename
        )

        # Ensure the expected requests were made against OPeNDAP.
        # See related comment in self.test_geo_bbox_end_to_end
        self.assertEqual(mock_util_download.call_count, 3)
        mock_util_download.assert_has_calls(
            [
                call(
                    f'{self.granule_url}.dmr.xml',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=None,
                    cfg=hoss.config,
                ),
                call(
                    f'{self.granule_url}.dap.nc4',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=ANY,
                    cfg=hoss.config,
                ),
                call(
                    f'{self.granule_url}.dap.nc4',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=ANY,
                    cfg=hoss.config,
                ),
            ]
        )

        # Ensure the constraint expression for dimensions data included only
        # geographic or temporal variables with no index ranges
        dimensions_data = mock_util_download.call_args_list[1][1].get('data', {})
        self.assert_valid_request_data(
            dimensions_data, {'%2Flatitude', '%2Flongitude', '%2Ftime'}
        )
        # Ensure the constraint expression contains all the required variables.
        # /wind_speed[][540:599][60:119], /time, /longitude[60:119]
        # /latitude[540:599]
        index_range_data = mock_util_download.call_args_list[2][1].get('data', {})
        self.assert_valid_request_data(
            index_range_data,
            {
                '%2Ftime',
                '%2Flatitude%5B540%3A599%5D',
                '%2Flongitude%5B60%3A119%5D',
                '%2Fwind_speed%5B%5D%5B540%3A599%5D%5B60%3A119%5D',
            },
        )

        # Ensure the output was staged with the expected file name
        mock_stage.assert_called_once_with(
            f'{self.tmp_dir}/uuid2.nc4',
            expected_output_basename,
            'application/x-netcdf4',
            location=self.staging_location,
            logger=hoss.logger,
        )
        mock_rmtree.assert_called_once_with(self.tmp_dir)

        # Ensure no variables were filled
        mock_get_fill_slice.assert_not_called()

    @patch('hoss.dimension_utilities.get_fill_slice')
    @patch('hoss.utilities.uuid4')
    @patch('hoss.adapter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('hoss.utilities.util_download')
    @patch('hoss.adapter.stage')
    def test_projected_grid_bbox(
        self,
        mock_stage,
        mock_util_download,
        mock_rmtree,
        mock_mkdtemp,
        mock_uuid,
        mock_get_fill_slice,
    ):
        """Make a request specifying a bounding box for a collection that is
        gridded to a non-geographic projection. This example will use
        ABoVE TVPRM, which uses an Albers Conical Equal Area projection
        with data covering Alaska.

        """
        expected_output_basename = 'opendap_url_NEE_subsetted.nc4'
        expected_staged_url = f'{self.staging_location}{expected_output_basename}'

        mock_uuid.side_effect = [Mock(hex='uuid'), Mock(hex='uuid2')]
        mock_mkdtemp.return_value = self.tmp_dir
        mock_stage.return_value = expected_staged_url

        dmr_path = 'tests/data/ABoVE_TVPRM_example.dmr'

        dimensions_path = f'{self.tmp_dir}/dimensions.nc4'
        copy('tests/data/ABoVE_TVPRM_prefetch.nc4', dimensions_path)

        output_path = f'{self.tmp_dir}/ABoVE_TVPRM_bbox.nc4'
        copy('tests/data/ABoVE_TVPRM_prefetch.nc4', output_path)
        mock_util_download.side_effect = [dmr_path, dimensions_path, output_path]

        message = Message(
            {
                'accessToken': 'fake-token',
                'callback': 'https://example.com/',
                'sources': [
                    {
                        'collection': 'C1234567890-EEDTEST',
                        'shortName': 'NorthSlope_NEE_TVPRM_1920',
                        'variables': [{'id': '', 'name': '/NEE', 'fullPath': '/NEE'}],
                    }
                ],
                'stagingLocation': self.staging_location,
                'subset': {'bbox': [-160, 68, -145, 70]},
                'user': 'wfunk',
            }
        )

        hoss = HossAdapter(message, config=config(False), catalog=self.input_stac)
        _, output_catalog = hoss.invoke()

        # Ensure that there is a single item in the output catalog with the
        # expected asset:
        self.assert_expected_output_catalog(
            output_catalog, expected_staged_url, expected_output_basename
        )

        # Ensure the expected requests were made against OPeNDAP.
        # See related comment in self.test_geo_bbox_end_to_end
        self.assertEqual(mock_util_download.call_count, 3)
        mock_util_download.assert_has_calls(
            [
                call(
                    f'{self.granule_url}.dmr.xml',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=None,
                    cfg=hoss.config,
                ),
                call(
                    f'{self.granule_url}.dap.nc4',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=ANY,
                    cfg=hoss.config,
                ),
                call(
                    f'{self.granule_url}.dap.nc4',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=ANY,
                    cfg=hoss.config,
                ),
            ]
        )

        # Ensure the constraint expression for dimensions data included only
        # spatial or temporal variables with no index ranges
        dimensions_data = mock_util_download.call_args_list[1][1].get('data', {})
        self.assert_valid_request_data(
            dimensions_data, {'%2Fx', '%2Fy', '%2Ftime', '%2Ftime_bnds'}
        )
        # Ensure the constraint expression contains all the required variables.
        # /NEE[][7:26][37:56], /time, /x[37:56], /y[7:26]
        index_range_data = mock_util_download.call_args_list[2][1].get('data', {})
        self.assert_valid_request_data(
            index_range_data,
            {
                '%2Ftime',
                '%2Ftime_bnds',
                '%2Fcrs',
                '%2Fx%5B37%3A56%5D',
                '%2Fy%5B7%3A26%5D',
                '%2FNEE%5B%5D%5B7%3A26%5D%5B37%3A56%5D',
            },
        )

        # Ensure the output was staged with the expected file name
        mock_stage.assert_called_once_with(
            f'{self.tmp_dir}/uuid2.nc4',
            expected_output_basename,
            'application/x-netcdf4',
            location=self.staging_location,
            logger=hoss.logger,
        )
        mock_rmtree.assert_called_once_with(self.tmp_dir)

        # Ensure no variables were filled
        mock_get_fill_slice.assert_not_called()

    @patch('hoss.dimension_utilities.get_fill_slice')
    @patch('hoss.utilities.uuid4')
    @patch('hoss.adapter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('hoss.bbox_utilities.download')
    @patch('hoss.utilities.util_download')
    @patch('hoss.adapter.stage')
    def test_projected_grid_shape(
        self,
        mock_stage,
        mock_util_download,
        mock_geojson_download,
        mock_rmtree,
        mock_mkdtemp,
        mock_uuid,
        mock_get_fill_slice,
    ):
        """Make a request specifying a shape file for a collection that is
        gridded to a non-geographic projection. This example will use
        ABoVE TVPRM, which uses an Albers Conical Equal Area projection
        with data covering Alaska.

        """
        expected_output_basename = 'opendap_url_NEE_subsetted.nc4'
        expected_staged_url = f'{self.staging_location}{expected_output_basename}'

        mock_uuid.side_effect = [Mock(hex='uuid'), Mock(hex='uuid2')]
        mock_mkdtemp.return_value = self.tmp_dir
        mock_stage.return_value = expected_staged_url

        dmr_path = 'tests/data/ABoVE_TVPRM_example.dmr'

        dimensions_path = f'{self.tmp_dir}/dimensions.nc4'
        copy('tests/data/ABoVE_TVPRM_prefetch.nc4', dimensions_path)

        geojson_path = f'{self.tmp_dir}/polygon.geo.json'
        copy('tests/geojson_examples/above_polygon.geo.json', geojson_path)

        shape_file_url = 'www.example.com/polygon.geo.json'
        mock_geojson_download.return_value = geojson_path

        output_path = f'{self.tmp_dir}/ABoVE_TVPRM_bbox.nc4'
        copy('tests/data/ABoVE_TVPRM_prefetch.nc4', output_path)
        mock_util_download.side_effect = [dmr_path, dimensions_path, output_path]

        message = Message(
            {
                'accessToken': 'fake-token',
                'callback': 'https://example.com/',
                'sources': [
                    {
                        'collection': 'C1234567890-EEDTEST',
                        'shortName': 'NorthSlope_NEE_TVPRM_1920',
                        'variables': [{'id': '', 'name': '/NEE', 'fullPath': '/NEE'}],
                    }
                ],
                'stagingLocation': self.staging_location,
                'subset': {
                    'shape': {'href': shape_file_url, 'type': 'application/geo+json'}
                },
                'user': 'wfunk',
            }
        )

        hoss = HossAdapter(message, config=config(False), catalog=self.input_stac)
        _, output_catalog = hoss.invoke()

        # Ensure that there is a single item in the output catalog with the
        # expected asset:
        self.assert_expected_output_catalog(
            output_catalog, expected_staged_url, expected_output_basename
        )

        # Ensure the expected requests were made against OPeNDAP.
        # See related comment in self.test_geo_bbox_end_to_end
        self.assertEqual(mock_util_download.call_count, 3)
        mock_util_download.assert_has_calls(
            [
                call(
                    f'{self.granule_url}.dmr.xml',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=None,
                    cfg=hoss.config,
                ),
                call(
                    f'{self.granule_url}.dap.nc4',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=ANY,
                    cfg=hoss.config,
                ),
                call(
                    f'{self.granule_url}.dap.nc4',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=ANY,
                    cfg=hoss.config,
                ),
            ]
        )

        # Ensure the constraint expression for dimensions data included only
        # geographic or temporal variables with no index ranges
        dimensions_data = mock_util_download.call_args_list[1][1].get('data', {})
        self.assert_valid_request_data(
            dimensions_data, {'%2Fx', '%2Fy', '%2Ftime', '%2Ftime_bnds'}
        )
        # Ensure the constraint expression contains all the required variables.
        # /NEE[][11:26][37:56], /time, /x[37:56], /y[11:26]
        index_range_data = mock_util_download.call_args_list[2][1].get('data', {})
        self.assert_valid_request_data(
            index_range_data,
            {
                '%2Ftime',
                '%2Ftime_bnds',
                '%2Fcrs',
                '%2Fx%5B37%3A56%5D',
                '%2Fy%5B11%3A26%5D',
                '%2FNEE%5B%5D%5B11%3A26%5D%5B37%3A56%5D',
            },
        )

        # Ensure the output was staged with the expected file name
        mock_stage.assert_called_once_with(
            f'{self.tmp_dir}/uuid2.nc4',
            expected_output_basename,
            'application/x-netcdf4',
            location='s3://example-bucket/',
            logger=hoss.logger,
        )
        mock_rmtree.assert_called_once_with(self.tmp_dir)

        # Ensure no variables were filled
        mock_get_fill_slice.assert_not_called()

    @patch('hoss.dimension_utilities.get_fill_slice')
    @patch('hoss.utilities.uuid4')
    @patch('hoss.adapter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('hoss.utilities.util_download')
    @patch('hoss.adapter.stage')
    def test_bounds_end_to_end(
        self,
        mock_stage,
        mock_util_download,
        mock_rmtree,
        mock_mkdtemp,
        mock_uuid,
        mock_get_fill_slice,
    ):
        """Ensure a request with a bounding box and temporal range will be
        correctly processed for a geographically gridded collection that
        has bounds variables for each dimension.

        Note: Each GPM IMERGHH granule has a single time slice, so the full
        range will be retrieved (e.g., /Grid/time[0:0]

        * -30.0 ≤ /Grid/lon[1500] ≤ -29.9
        * 45.0 ≤ /Grid/lat[1350] ≤ 45.1
        * -14.9 ≤ /Grid/lon[1649] ≤ -15.0
        * 59.9 ≤ /Grid/lat[1499] ≤ 60.0

        """
        expected_output_basename = 'opendap_url_Grid_precipitationCal_subsetted.nc4'
        expected_staged_url = f'{self.staging_location}{expected_output_basename}'

        mock_uuid.side_effect = [Mock(hex='uuid'), Mock(hex='uuid2')]
        mock_mkdtemp.return_value = self.tmp_dir
        mock_stage.return_value = expected_staged_url

        dmr_path = write_dmr(self.tmp_dir, self.gpm_imerghh_dmr)

        dimensions_path = f'{self.tmp_dir}/dimensions.nc4'
        copy('tests/data/GPM_3IMERGHH_prefetch.nc4', dimensions_path)

        all_variables_path = f'{self.tmp_dir}/variables.nc4'
        copy('tests/data/GPM_3IMERGHH_bounds.nc4', all_variables_path)

        mock_util_download.side_effect = [dmr_path, dimensions_path, all_variables_path]

        message = Message(
            {
                'accessToken': 'fake-token',
                'callback': 'https://example.com/',
                'sources': [
                    {
                        'collection': 'C1234567890-EEDTEST',
                        'shortName': 'GPM_3IMERGHH',
                        'variables': [
                            {
                                'id': '',
                                'name': self.gpm_variable,
                                'fullPath': self.gpm_variable,
                            }
                        ],
                    }
                ],
                'stagingLocation': self.staging_location,
                'subset': {'bbox': [-30, 45, -15, 60]},
                'temporal': {
                    'start': '2020-01-01T12:15:00',
                    'end': '2020-01-01T12:45:00',
                },
                'user': 'jlovell',
            }
        )

        hoss = HossAdapter(message, config=config(False), catalog=self.input_stac)
        _, output_catalog = hoss.invoke()

        # Ensure that there is a single item in the output catalog with the
        # expected asset:
        self.assert_expected_output_catalog(
            output_catalog, expected_staged_url, expected_output_basename
        )

        # Ensure the expected requests were made against OPeNDAP.
        # See related comment in self.test_geo_bbox_end_to_end
        self.assertEqual(mock_util_download.call_count, 3)
        mock_util_download.assert_has_calls(
            [
                call(
                    f'{self.granule_url}.dmr.xml',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=None,
                    cfg=hoss.config,
                ),
                call(
                    f'{self.granule_url}.dap.nc4',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=ANY,
                    cfg=hoss.config,
                ),
                call(
                    f'{self.granule_url}.dap.nc4',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=ANY,
                    cfg=hoss.config,
                ),
            ]
        )

        # Ensure the constraint expression for dimensions data included only
        # dimension variables and their associated bounds variables.
        dimensions_data = mock_util_download.call_args_list[1][1].get('data', {})
        self.assert_valid_request_data(
            dimensions_data,
            {
                '%2FGrid%2Flat',
                '%2FGrid%2Flat_bnds',
                '%2FGrid%2Flon',
                '%2FGrid%2Flon_bnds',
                '%2FGrid%2Ftime',
                '%2FGrid%2Ftime_bnds',
            },
        )
        # Ensure the constraint expression contains all the required variables.
        # /Grid/precipitationCal[0:0][1500:1649][1350:1499],
        # /Grid/time[0:0], /Grid/time_bnds[0:0][]
        # /Grid/lat[1350:1499], /Grid/lat_bnds[1350:1499][],
        # /Grid/lon[1500:1649], /Grid/lon_bnds[1500:1649][]
        index_range_data = mock_util_download.call_args_list[2][1].get('data', {})
        self.assert_valid_request_data(
            index_range_data,
            {
                '%2FGrid%2Flat%5B1350%3A1499%5D',
                '%2FGrid%2Flat_bnds%5B1350%3A1499%5D%5B%5D',
                '%2FGrid%2Flon%5B1500%3A1649%5D',
                '%2FGrid%2Flon_bnds%5B1500%3A1649%5D%5B%5D',
                '%2FGrid%2Ftime%5B0%3A0%5D',
                '%2FGrid%2Ftime_bnds%5B0%3A0%5D%5B%5D',
                '%2FGrid%2FprecipitationCal%5B0%3A0%5D%5B1500%3A1649%5D%5B1350%3A1499%5D',
            },
        )

        # Ensure the output was staged with the expected file name
        mock_stage.assert_called_once_with(
            f'{self.tmp_dir}/uuid2.nc4',
            expected_output_basename,
            'application/x-netcdf4',
            location=self.staging_location,
            logger=hoss.logger,
        )
        mock_rmtree.assert_called_once_with(self.tmp_dir)

        # Ensure no variables were filled
        mock_get_fill_slice.assert_not_called()

    @patch('hoss.dimension_utilities.get_fill_slice')
    @patch('hoss.utilities.uuid4')
    @patch('hoss.adapter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('hoss.utilities.util_download')
    @patch('hoss.adapter.stage')
    def test_requested_dimensions_bounds_end_to_end(
        self,
        mock_stage,
        mock_util_download,
        mock_rmtree,
        mock_mkdtemp,
        mock_uuid,
        mock_get_fill_slice,
    ):
        """Ensure a request with a spatial range specified by variable names,
        not just subset=lat(), subset=lon(), will be correctly processed
        for a geographically gridded collection that has bounds variables
        for each dimension.

        Note: Each GPM IMERGHH granule has a single time slice, so the full
        range will be retrieved (e.g., /Grid/time[0:0]

        * -30.0 ≤ /Grid/lon[1500] ≤ -29.9
        * 45.0 ≤ /Grid/lat[1350] ≤ 45.1
        * -14.9 ≤ /Grid/lon[1649] ≤ -15.0
        * 59.9 ≤ /Grid/lat[1499] ≤ 60.0

        """
        expected_output_basename = 'opendap_url_Grid_precipitationCal_subsetted.nc4'
        expected_staged_url = ''.join([self.staging_location, expected_output_basename])

        mock_uuid.side_effect = [Mock(hex='uuid'), Mock(hex='uuid2')]
        mock_mkdtemp.return_value = self.tmp_dir
        mock_stage.return_value = expected_staged_url

        dmr_path = write_dmr(self.tmp_dir, self.gpm_imerghh_dmr)

        dimensions_path = f'{self.tmp_dir}/dimensions.nc4'
        copy('tests/data/GPM_3IMERGHH_prefetch.nc4', dimensions_path)

        all_variables_path = f'{self.tmp_dir}/variables.nc4'
        copy('tests/data/GPM_3IMERGHH_bounds.nc4', all_variables_path)

        mock_util_download.side_effect = [dmr_path, dimensions_path, all_variables_path]

        message = Message(
            {
                'accessToken': 'fake-token',
                'callback': 'https://example.com/',
                'sources': [
                    {
                        'collection': 'C1234567890-EEDTEST',
                        'shortName': 'GPM_3IMERGHH',
                        'variables': [
                            {
                                'id': '',
                                'name': self.gpm_variable,
                                'fullPath': self.gpm_variable,
                            }
                        ],
                    }
                ],
                'stagingLocation': self.staging_location,
                'subset': {
                    'dimensions': [
                        {'name': '/Grid/lat', 'min': 45, 'max': 60},
                        {'name': '/Grid/lon', 'min': -30, 'max': -15},
                    ]
                },
                'temporal': {
                    'start': '2020-01-01T12:15:00',
                    'end': '2020-01-01T12:45:00',
                },
                'user': 'jlovell',
            }
        )

        hoss = HossAdapter(message, config=config(False), catalog=self.input_stac)
        _, output_catalog = hoss.invoke()

        # Ensure that there is a single item in the output catalog with the
        # expected asset:
        self.assert_expected_output_catalog(
            output_catalog, expected_staged_url, expected_output_basename
        )

        # Ensure the expected requests were made against OPeNDAP.
        # See related comment in self.test_geo_bbox_end_to_end
        self.assertEqual(mock_util_download.call_count, 3)
        mock_util_download.assert_has_calls(
            [
                call(
                    f'{self.granule_url}.dmr.xml',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=None,
                    cfg=hoss.config,
                ),
                call(
                    f'{self.granule_url}.dap.nc4',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=ANY,
                    cfg=hoss.config,
                ),
                call(
                    f'{self.granule_url}.dap.nc4',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=ANY,
                    cfg=hoss.config,
                ),
            ]
        )

        # Ensure the constraint expression for dimensions data included only
        # dimension variables and their associated bounds variables.
        dimensions_data = mock_util_download.call_args_list[1][1].get('data', {})
        self.assert_valid_request_data(
            dimensions_data,
            {
                '%2FGrid%2Flat',
                '%2FGrid%2Flat_bnds',
                '%2FGrid%2Flon',
                '%2FGrid%2Flon_bnds',
                '%2FGrid%2Ftime',
                '%2FGrid%2Ftime_bnds',
            },
        )
        # Ensure the constraint expression contains all the required variables.
        # /Grid/precipitationCal[0:0][1500:1649][1350:1499],
        # /Grid/time[0:0], /Grid/time_bnds[0:0][]
        # /Grid/lat[1350:1499], /Grid/lat_bnds[1350:1499][],
        # /Grid/lon[1500:1649], /Grid/lon_bnds[1500:1649][]
        index_range_data = mock_util_download.call_args_list[2][1].get('data', {})
        self.assert_valid_request_data(
            index_range_data,
            {
                '%2FGrid%2Flat%5B1350%3A1499%5D',
                '%2FGrid%2Flat_bnds%5B1350%3A1499%5D%5B%5D',
                '%2FGrid%2Flon%5B1500%3A1649%5D',
                '%2FGrid%2Flon_bnds%5B1500%3A1649%5D%5B%5D',
                '%2FGrid%2Ftime%5B0%3A0%5D',
                '%2FGrid%2Ftime_bnds%5B0%3A0%5D%5B%5D',
                '%2FGrid%2FprecipitationCal%5B0%3A0%5D%5B1500%3A1649%5D%5B1350%3A1499%5D',
            },
        )

        # Ensure the output was staged with the expected file name
        mock_stage.assert_called_once_with(
            f'{self.tmp_dir}/uuid2.nc4',
            expected_output_basename,
            'application/x-netcdf4',
            location=self.staging_location,
            logger=hoss.logger,
        )
        mock_rmtree.assert_called_once_with(self.tmp_dir)

        # Ensure no variables were filled
        mock_get_fill_slice.assert_not_called()

    @patch('hoss.adapter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('hoss.subset.download_url')
    @patch('hoss.adapter.stage')
    def test_retriable_exception_handling(
        self, mock_stage, mock_download_subset, mock_rmtree, mock_mkdtemp
    ):
        """Ensure that if a retriable exception is raised during processing, this
        causes a HarmonyException to be raised, to allow for informative
        logging.

        """
        mock_mkdtemp.return_value = self.tmp_dir
        mock_download_subset.side_effect = Exception('Random error')

        message = Message(
            {
                'accessToken': 'fake-token',
                'callback': 'https://example.com/',
                'sources': [
                    {
                        'collection': 'C1234567890-EEDTEST',
                        'shortName': 'ATL03',
                        'variables': [
                            {
                                'id': '',
                                'name': self.atl03_variable,
                                'fullPath': self.atl03_variable,
                            }
                        ],
                    }
                ],
                'stagingLocation': self.staging_location,
                'user': 'kmattingly',
            }
        )

        with self.assertRaises(HarmonyException):
            hoss = HossAdapter(message, config=config(False), catalog=self.input_stac)
            hoss.invoke()

        mock_stage.assert_not_called()
        mock_rmtree.assert_called_once_with(self.tmp_dir)

    @patch('hoss.adapter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('hoss.subset.check_invalid_variable_request')
    @patch('hoss.adapter.stage')
    def test_not_retriable_exception_handling(
        self, mock_stage, mock_check_variable, mock_rmtree, mock_mkdtemp
    ):
        """Ensure that if a not retriable exception is raised during
        processing, this causes a NoRetryException to be raised, to prevent
        extra Harmony CPU cycles.

        """
        mock_mkdtemp.return_value = self.tmp_dir
        mock_check_variable.side_effect = InvalidVariableRequest('Random error')

        message = Message(
            {
                'accessToken': 'fake-token',
                'callback': 'https://example.com/',
                'sources': [{}],
            }
        )

        with self.assertRaises(NoRetryException):
            hoss = HossAdapter(message, config=config(False), catalog=self.input_stac)
            hoss.invoke()

        mock_stage.assert_not_called()
        mock_rmtree.assert_called_once_with(self.tmp_dir)

    @patch('hoss.dimension_utilities.get_fill_slice')
    @patch('hoss.utilities.uuid4')
    @patch('hoss.adapter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('hoss.utilities.util_download')
    @patch('hoss.adapter.stage')
    def test_edge_aligned_no_bounds_end_to_end(
        self,
        mock_stage,
        mock_util_download,
        mock_rmtree,
        mock_mkdtemp,
        mock_uuid,
        mock_get_fill_slice,
    ):
        """Ensure a request for a collection that contains dimension variables
        with edge-aligned grid cells is correctly processed regardless of
        whether or not a bounds variable associated with that dimension
        variable exists.

        """
        expected_output_basename = 'opendap_url_global_asr_obs_grid_subsetted.nc4'
        expected_staged_url = f'{self.staging_location}{expected_output_basename}'

        mock_uuid.side_effect = [Mock(hex='uuid'), Mock(hex='uuid2')]
        mock_mkdtemp.return_value = self.tmp_dir
        mock_stage.return_value = expected_staged_url

        dmr_path = write_dmr(self.tmp_dir, self.atl16_dmr)

        dimensions_path = f'{self.tmp_dir}/dimensions.nc4'
        copy('tests/data/ATL16_prefetch.nc4', dimensions_path)

        all_variables_path = f'{self.tmp_dir}/variables.nc4'
        copy('tests/data/ATL16_variables.nc4', all_variables_path)

        mock_util_download.side_effect = [dmr_path, dimensions_path, all_variables_path]

        message = Message(
            {
                'accessToken': 'fake-token',
                'callback': 'https://example.com/',
                'sources': [
                    {
                        'collection': 'C1238589498-EEDTEST',
                        'shortName': 'ATL16',
                        'variables': [
                            {
                                'id': '',
                                'name': self.atl16_variable,
                                'fullPath': self.atl16_variable,
                            }
                        ],
                    }
                ],
                'stagingLocation': self.staging_location,
                'subset': {'bbox': [77, 71.25, 88, 74.75]},
                'user': 'sride',
            }
        )

        hoss = HossAdapter(message, config=config(False), catalog=self.input_stac)
        _, output_catalog = hoss.invoke()

        # Ensure that there is a single item in the output catalog with the
        # expected asset:
        self.assert_expected_output_catalog(
            output_catalog, expected_staged_url, expected_output_basename
        )

        # Ensure the expected requests were made against OPeNDAP.
        self.assertEqual(mock_util_download.call_count, 3)
        mock_util_download.assert_has_calls(
            [
                call(
                    f'{self.granule_url}.dmr.xml',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=None,
                    cfg=hoss.config,
                ),
                call(
                    f'{self.granule_url}.dap.nc4',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=ANY,
                    cfg=hoss.config,
                ),
                call(
                    f'{self.granule_url}.dap.nc4',
                    self.tmp_dir,
                    hoss.logger,
                    access_token=message.accessToken,
                    data=ANY,
                    cfg=hoss.config,
                ),
            ]
        )

        # Ensure the constraint expression for dimensions data included only
        # dimension variables and their associated bounds variables.
        dimensions_data = mock_util_download.call_args_list[1][1].get('data', {})
        self.assert_valid_request_data(
            dimensions_data, {'%2Fglobal_grid_lat', '%2Fglobal_grid_lon'}
        )

        # Ensure the constraint expression contains all the required variables.
        # The latitude and longitude index ranges here depend on whether
        # the cells have centre-alignment or edge-alignment.
        # Previously, the incorrect index ranges assuming centre-alignment:
        #   latitude [54:55] with values (72,75)
        #   longitude [86:89] with values (78,81,84,87)
        #
        # Now, the correct index ranges with edge-alignment:
        #   latitude: [53:54] for values (69,72).
        #   longitude:[85:89] for values (75,78,81,84,87)
        #
        index_range_data = mock_util_download.call_args_list[2][1].get('data', {})
        self.assert_valid_request_data(
            index_range_data,
            {
                '%2Fglobal_asr_obs_grid%5B53%3A54%5D%5B85%3A89%5D',
                '%2Fglobal_grid_lat%5B53%3A54%5D',
                '%2Fglobal_grid_lon%5B85%3A89%5D',
            },
        )

        # Ensure the output was staged with the expected file name
        mock_stage.assert_called_once_with(
            f'{self.tmp_dir}/uuid2.nc4',
            expected_output_basename,
            'application/x-netcdf4',
            location=self.staging_location,
            logger=hoss.logger,
        )

        mock_rmtree.assert_called_once_with(self.tmp_dir)

        # Ensure no variables were filled
        mock_get_fill_slice.assert_not_called()

    @patch('hoss.adapter.mkdtemp')
    @patch('shutil.rmtree')
    @patch('hoss.adapter.stage')
    @patch('hoss.subset.get_prefetch_variables')
    @patch('hoss.subset.get_varinfo')
    def test_no_data_exception_handling(
        self,
        mock_get_varinfo,
        mock_get_prefetch_variables,
        mock_stage,
        mock_rmtree,
        mock_mkdtemp,
    ):
        """Ensure that if a NoDataException is raised during processing, this
        is captured in the adapter and output is not staged.

        """
        mock_mkdtemp.return_value = self.tmp_dir
        smap_varinfo = VarInfoFromDmr(
            'tests/data/SC_SPL3FTP_004.dmr',
            'SPL3FTP',
            'hoss/hoss_config.json',
        )
        prefetch_path = 'tests/data/SC_SPL3FTP_004_Polar_prefetch.nc4'
        mock_get_varinfo.return_value = smap_varinfo
        mock_get_prefetch_variables.return_value = prefetch_path

        message = Message(
            {
                'accessToken': 'fake-token',
                'callback': 'https://example.com/',
                'sources': [
                    {
                        'collection': 'C1268617120-EEDTEST',
                        'shortName': 'SPL3FTP',
                        'variables': [
                            {
                                'id': 'V1247777461-EEDTEST',
                                'name': 'surface_flag',
                                'fullPath': '/Freeze_Thaw_Retrieval_Data_Polar/surface_flag',
                            },
                        ],
                    }
                ],
                'subset': {'bbox': [-179.9, -89.8, -179.8, -89.5]},
                'stagingLocation': self.staging_location,
                'user': 'testuser',
            }
        )
        with self.assertRaises(NoDataException):
            hoss = HossAdapter(message, config=config(False), catalog=self.input_stac)
            hoss.invoke()

        mock_stage.assert_not_called()
        mock_rmtree.assert_called_once_with(self.tmp_dir)
