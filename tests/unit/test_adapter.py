from typing import List, Dict, Optional
from unittest import TestCase
from unittest.mock import patch, ANY
import json

from harmony.message import Message
from harmony.util import config

from hoss.adapter import HossAdapter
from hoss.bbox_utilities import BBox
from tests.utilities import create_stac, Granule, spy_on


@patch('hoss.adapter.get_file_mimetype')
@patch('hoss.adapter.subset_granule')
@patch('hoss.adapter.stage')
class TestAdapter(TestCase):
    """Test the HossAdapter class for basic functionality including:

    - Synchronous vs asynchronous behaviour.
    - Basic message validation.

    """

    @classmethod
    def setUpClass(cls):
        cls.operations = {
            'is_variable_subset': True,
            'is_regridded': False,
            'is_subsetted': False,
        }
        cls.africa_granule_url = '/home/tests/data/africa.nc'
        cls.africa_stac = create_stac(
            [Granule(cls.africa_granule_url, None, ['opendap', 'data'])]
        )

    def setUp(self):
        self.config = config(validate=False)
        self.process_item_spy = spy_on(HossAdapter.process_item)

    def create_message(
        self,
        collection_id: str,
        collection_short_name: str,
        variable_list: List[str],
        user: str,
        is_synchronous: Optional[bool] = None,
        bounding_box: Optional[List[float]] = None,
        temporal_range: Optional[Dict[str, str]] = None,
        shape_file: Optional[str] = None,
        dimensions: Optional[List[Dict]] = None,
    ) -> Message:
        """Create a Harmony Message object with the requested attributes."""
        variables = [{'name': variable} for variable in variable_list]
        message_content = {
            'sources': [
                {
                    'collection': collection_id,
                    'shortName': collection_short_name,
                    'variables': variables,
                }
            ],
            'user': user,
            'callback': 'https://example.com/',
            'stagingLocation': 's3://example-bucket/',
            'accessToken': 'xyzzy',
            'subset': {'bbox': bounding_box, 'dimensions': dimensions, 'shape': None},
            'temporal': temporal_range,
        }

        if shape_file is not None:
            message_content['subset']['shape'] = {
                'href': shape_file,
                'type': 'application/geo+json',
            }

        if is_synchronous is not None:
            message_content['isSynchronous'] = is_synchronous

        return Message(json.dumps(message_content))

    def test_temporal_request(self, mock_stage, mock_subset_granule, mock_get_mimetype):
        """A request that specifies a temporal range should result in a
        temporal subset.

        """
        mock_subset_granule.return_value = '/path/to/output.nc'
        mock_get_mimetype.return_value = ('application/x-netcdf4', None)
        temporal_range = {'start': '2021-01-01T00:00:00', 'end': '2021-01-02T00:00:00'}
        collection_short_name = 'harmony_example_l2'

        message = self.create_message(
            'C1233860183-EEDTEST',
            collection_short_name,
            ['alpha_var', 'blue_var'],
            'mcollins',
            bounding_box=None,
            temporal_range=temporal_range,
        )

        hoss = HossAdapter(message, config=self.config, catalog=self.africa_stac)

        with patch.object(HossAdapter, 'process_item', self.process_item_spy):
            hoss.invoke()

        mock_subset_granule.assert_called_once_with(
            self.africa_granule_url,
            message.sources[0],
            ANY,
            hoss.message,
            hoss.logger,
            hoss.config,
        )
        mock_get_mimetype.assert_called_once_with('/path/to/output.nc')

        mock_stage.assert_called_once_with(
            '/path/to/output.nc',
            'africa_subsetted.nc4',
            'application/x-netcdf4',
            location='s3://example-bucket/',
            logger=hoss.logger,
        )

    def test_synchronous_request(
        self, mock_stage, mock_subset_granule, mock_get_mimetype
    ):
        """A request that specifies `isSynchronous = True` should complete
        for a single granule. It should call the `subset_granule` function,
        and then indicate the request completed.

        """
        mock_subset_granule.return_value = '/path/to/output.nc'
        mock_get_mimetype.return_value = ('application/x-netcdf4', None)
        collection_short_name = 'harmony_example_l2'

        message = self.create_message(
            'C1233860183-EEDTEST',
            collection_short_name,
            ['alpha_var', 'blue_var'],
            'narmstrong',
            True,
        )

        hoss = HossAdapter(message, config=self.config, catalog=self.africa_stac)

        with patch.object(HossAdapter, 'process_item', self.process_item_spy):
            hoss.invoke()

        mock_subset_granule.assert_called_once_with(
            self.africa_granule_url,
            message.sources[0],
            ANY,
            hoss.message,
            hoss.logger,
            hoss.config,
        )

        mock_get_mimetype.assert_called_once_with('/path/to/output.nc')

        mock_stage.assert_called_once_with(
            '/path/to/output.nc',
            'africa_subsetted.nc4',
            'application/x-netcdf4',
            location='s3://example-bucket/',
            logger=hoss.logger,
        )

    def test_asynchronous_request(
        self, mock_stage, mock_subset_granule, mock_get_mimetype
    ):
        """A request that specified `isSynchronous = False` should complete
        for a single granule. It should call the `subset_granule` function,
        and then indicate the request completed.

        """
        mock_subset_granule.return_value = '/path/to/output.nc'
        mock_get_mimetype.return_value = ('application/x-netcdf4', None)
        collection_short_name = 'harmony_example_l2'

        message = self.create_message(
            'C1233860183-EEDTEST',
            collection_short_name,
            ['alpha_var', 'blue_var'],
            'ealdrin',
            False,
        )

        hoss = HossAdapter(message, config=self.config, catalog=self.africa_stac)

        with patch.object(HossAdapter, 'process_item', self.process_item_spy):
            hoss.invoke()

        mock_subset_granule.assert_called_once_with(
            self.africa_granule_url,
            message.sources[0],
            ANY,
            hoss.message,
            hoss.logger,
            hoss.config,
        )
        mock_get_mimetype.assert_called_once_with('/path/to/output.nc')

        mock_stage.assert_called_once_with(
            '/path/to/output.nc',
            'africa_subsetted.nc4',
            'application/x-netcdf4',
            location='s3://example-bucket/',
            logger=hoss.logger,
        )

    def test_unspecified_synchronous_request(
        self, mock_stage, mock_subset_granule, mock_get_mimetype
    ):
        """A request the does not specify `isSynchronous` should default to
        synchronous behaviour. The `subset_granule` function should be
        called. Then the request should complete.

        """
        mock_subset_granule.return_value = '/path/to/output.nc'
        mock_get_mimetype.return_value = ('application/x-netcdf4', None)
        collection_short_name = 'harmony_example_l2'

        message = self.create_message(
            'C1233860183-EEDTEST',
            collection_short_name,
            ['alpha_var', 'blue_var'],
            'mcollins',
        )

        hoss = HossAdapter(message, config=self.config, catalog=self.africa_stac)

        with patch.object(HossAdapter, 'process_item', self.process_item_spy):
            hoss.invoke()

        mock_subset_granule.assert_called_once_with(
            self.africa_granule_url,
            message.sources[0],
            ANY,
            hoss.message,
            hoss.logger,
            hoss.config,
        )
        mock_get_mimetype.assert_called_once_with('/path/to/output.nc')

        mock_stage.assert_called_once_with(
            '/path/to/output.nc',
            'africa_subsetted.nc4',
            'application/x-netcdf4',
            location='s3://example-bucket/',
            logger=hoss.logger,
        )

    def test_hoss_bbox_request(
        self, mock_stage, mock_subset_granule, mock_get_mimetype
    ):
        """A request that specifies a bounding box should result in a both a
        variable and a bounding box spatial subset being made.

        """
        mock_subset_granule.return_value = '/path/to/output.nc'
        mock_get_mimetype.return_value = ('application/x-netcdf4', None)
        bounding_box = BBox(-20, -10, 20, 30)
        collection_short_name = 'harmony_example_l2'

        message = self.create_message(
            'C1233860183-EEDTEST',
            collection_short_name,
            ['alpha_var', 'blue_var'],
            'mcollins',
            bounding_box=bounding_box,
        )

        hoss = HossAdapter(message, config=self.config, catalog=self.africa_stac)

        with patch.object(HossAdapter, 'process_item', self.process_item_spy):
            hoss.invoke()

        mock_subset_granule.assert_called_once_with(
            self.africa_granule_url,
            message.sources[0],
            ANY,
            hoss.message,
            hoss.logger,
            hoss.config,
        )
        mock_get_mimetype.assert_called_once_with('/path/to/output.nc')

        mock_stage.assert_called_once_with(
            '/path/to/output.nc',
            'africa_subsetted.nc4',
            'application/x-netcdf4',
            location='s3://example-bucket/',
            logger=hoss.logger,
        )

    def test_hoss_shape_file_request(
        self, mock_stage, mock_subset_granule, mock_get_mimetype
    ):
        """A request that specifies a shape file should result in a both a
        variable and a spatial subset being made.

        """
        collection_short_name = 'harmony_example_l2'
        shape_file_url = 'www.example.com/shape.geo.json'
        mock_subset_granule.return_value = '/path/to/output.nc'
        mock_get_mimetype.return_value = ('application/x-netcdf4', None)

        message = self.create_message(
            'C1233860183-EEDTEST',
            collection_short_name,
            ['alpha_var', 'blue_var'],
            'mcollins',
            shape_file=shape_file_url,
        )

        hoss = HossAdapter(message, config=self.config, catalog=self.africa_stac)

        with patch.object(HossAdapter, 'process_item', self.process_item_spy):
            hoss.invoke()

        mock_subset_granule.assert_called_once_with(
            self.africa_granule_url,
            message.sources[0],
            ANY,
            hoss.message,
            hoss.logger,
            hoss.config,
        )
        mock_get_mimetype.assert_called_once_with('/path/to/output.nc')

        mock_stage.assert_called_once_with(
            '/path/to/output.nc',
            'africa_subsetted.nc4',
            'application/x-netcdf4',
            location='s3://example-bucket/',
            logger=hoss.logger,
        )

    def test_hoss_named_dimension(
        self, mock_stage, mock_subset_granule, mock_get_mimetype
    ):
        """A request with a message that specifies a named dimension within a
        granule, with a specific range of data, should have that
        information extracted from the input message and passed along to
        the `subset_granule` function.

        This unit test refers to a file that is not actually stored in the
        repository, as it would be large.

        """
        collection_short_name = 'M2I3NPASM'
        granule_url = '/home/tests/data/M2I3NPASM.nc4'
        mock_subset_granule.return_value = '/path/to/output.nc'
        mock_get_mimetype.return_value = ('application/x-netcdf4', None)

        message = self.create_message(
            'C1245663527-EEDTEST',
            collection_short_name,
            ['H1000'],
            'dbowman',
            dimensions=[{'name': 'lev', 'min': 800, 'max': 1000}],
        )
        input_stac = create_stac([Granule(granule_url, None, ['opendap', 'data'])])

        hoss = HossAdapter(message, config=self.config, catalog=input_stac)

        with patch.object(HossAdapter, 'process_item', self.process_item_spy):
            hoss.invoke()

        mock_subset_granule.assert_called_once_with(
            granule_url, message.sources[0], ANY, hoss.message, hoss.logger, hoss.config
        )
        mock_get_mimetype.assert_called_once_with('/path/to/output.nc')

        mock_stage.assert_called_once_with(
            '/path/to/output.nc',
            'M2I3NPASM_H1000_subsetted.nc4',
            'application/x-netcdf4',
            location='s3://example-bucket/',
            logger=hoss.logger,
        )

    def test_missing_granules(self, mock_stage, mock_subset_granule, mock_get_mimetype):
        """A request with no specified granules in an inbound Harmony message
        should raise an exception.

        """
        collection_short_name = 'harmony_example_l2'
        mock_subset_granule.return_value = '/path/to/output.nc'
        mock_get_mimetype.return_value = ('application/x-netcdf4', None)

        message = self.create_message(
            'C1233860183-EEDTEST',
            collection_short_name,
            ['alpha_var', 'blue_var'],
            'pconrad',
            False,
        )

        hoss = HossAdapter(message, config=self.config, catalog=create_stac([]))

        with self.assertRaises(Exception) as context_manager:
            with patch.object(HossAdapter, 'process_item', self.process_item_spy):
                hoss.invoke()

        self.assertEqual(
            str(context_manager.exception),
            'No granules specified for variable subsetting',
        )

        mock_subset_granule.assert_not_called()
        mock_get_mimetype.assert_not_called()

        mock_stage.assert_not_called()

    def test_asynchronous_multiple_granules(
        self, mock_stage, mock_subset_granule, mock_get_mimetype
    ):
        """A request for asynchronous processing, with multiple granules
        specified should be successful, and call `subset_granule` for each
        input granule.

        """
        output_paths = ['/path/to/output1.nc', '/path/to/output2.nc']
        output_filenames = [
            'africa_subsetted.nc4',
            'f16_ssmis_20200102v7_subsetted.nc4',
        ]

        mock_subset_granule.side_effect = output_paths
        mock_get_mimetype.return_value = ('application/x-netcdf4', None)
        collection_short_name = 'harmony_example_l2'

        message = self.create_message(
            'C1233860183-EEDTEST',
            collection_short_name,
            ['alpha_var', 'blue_var'],
            'abean',
            False,
        )
        input_stac = create_stac(
            [
                Granule(self.africa_granule_url, None, ['opendap', 'data']),
                Granule(
                    '/home/tests/data/f16_ssmis_20200102v7.nc',
                    None,
                    ['opendap', 'data'],
                ),
            ]
        )

        hoss = HossAdapter(message, config=self.config, catalog=input_stac)

        with patch.object(HossAdapter, 'process_item', self.process_item_spy):
            hoss.invoke()

        granules = hoss.message.granules

        for index, granule in enumerate(granules):
            mock_subset_granule.assert_any_call(
                granule.url,
                message.sources[0],
                ANY,
                hoss.message,
                hoss.logger,
                self.config,
            )
            mock_get_mimetype.assert_any_call(output_paths[index])

            mock_stage.assert_any_call(
                output_paths[index],
                output_filenames[index],
                'application/x-netcdf4',
                location=message.stagingLocation,
                logger=hoss.logger,
            )

    def test_missing_variables(
        self, mock_stage, mock_subset_granule, mock_get_mimetype
    ):
        """Ensure that if no variables are specified for a source, the service
        will not raise an exception, and that the variables specified to
        the `subset_granule` function is an empty list. The output of that
        function should be staged by Harmony.

        """
        mock_subset_granule.return_value = '/path/to/output.nc'
        mock_get_mimetype.return_value = ('application/x-netcdf4', None)
        collection_short_name = 'harmony_example_l2'

        message = self.create_message(
            'C1233860183-EEDTEST', collection_short_name, [], 'jlovell'
        )

        hoss = HossAdapter(message, config=self.config, catalog=self.africa_stac)

        with patch.object(HossAdapter, 'process_item', self.process_item_spy):
            hoss.invoke()

        mock_subset_granule.assert_called_once_with(
            self.africa_granule_url,
            message.sources[0],
            ANY,
            hoss.message,
            hoss.logger,
            hoss.config,
        )
        mock_get_mimetype.assert_called_once_with('/path/to/output.nc')

        mock_stage.assert_called_once_with(
            '/path/to/output.nc',
            'africa.nc4',
            'application/x-netcdf4',
            location='s3://example-bucket/',
            logger=hoss.logger,
        )
