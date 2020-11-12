from typing import List, Optional
from unittest import TestCase
from unittest.mock import patch, ANY
import json

from harmony.message import Message

from subsetter import HarmonyAdapter
from tests.utilities import contains, spy_on


@patch('subsetter.get_file_mimetype')
@patch('subsetter.subset_granule')
@patch('harmony.util.stage')
class TestSubsetter(TestCase):
    """ Test the HarmonyAdapter class for basic functionality including:

        - Synchronous vs asynchronous behaviour.
        - Basic message validation.

    """

    @classmethod
    def setUpClass(cls):
        cls.operations = {'is_variable_subset': True,
                          'is_regridded': False,
                          'is_subsetted': False}

    def setUp(self):
        self.process_item_spy = spy_on(HarmonyAdapter.process_item)

    def create_message(self, collection: str, granule_id: str, file_paths: List[str],
                       variable_list: List[str], user: str,
                       is_synchronous: Optional[bool] = None) -> Message:
        """ Create a Harmony Message object with the requested attributes. """
        granules = [
            {
                'id': granule_id,
                'url': file_path,
                'temporal': {
                    'start': '2020-01-01T00:00:00.000Z',
                    'end': '2020-01-02T00:00:00.000Z'
                },
                'bbox': [-180, -90, 180, 90]
            } for file_path in file_paths]
        variables = [{'name': variable} for variable in variable_list]
        message_content = {'sources': [{'collection': collection,
                                        'granules': granules,
                                        'variables': variables}],
                           'callback': 'https://example.com/',
                           'stagingLocation': 's3://example-bucket/',
                           'user': user}

        if is_synchronous is not None:
            message_content['isSynchronous'] = is_synchronous

        return Message(json.dumps(message_content))

    def test_synchronous_request(self,
                                 mock_stage,
                                 mock_subset_granule,
                                 mock_get_mimetype):
        """ A request that specifies `isSynchronous = True` should complete
            for a single granule. It should call the `subset_granule` function,
            and then indicate the request completed.

        """
        mock_subset_granule.return_value = '/path/to/output.nc'
        mock_get_mimetype.return_value = ('application/x-netcdf4', None)

        message = self.create_message('C1233860183-EEDTEST',
                                      'G1233860471-EEDTEST',
                                      ['/home/tests/data/africa.nc'],
                                      ['alpha_var', 'blue_var'],
                                      'narmstrong',
                                      True)
        variable_subsetter = HarmonyAdapter(message)
        with patch.object(HarmonyAdapter, 'process_item', self.process_item_spy):
            variable_subsetter.invoke()
        granule = variable_subsetter.message.granules[0]

        mock_subset_granule.assert_called_once_with(granule.url,
                                                    granule.variables,
                                                    ANY,
                                                    variable_subsetter.logger)
        mock_get_mimetype.assert_called_once_with('/path/to/output.nc')

        mock_stage.assert_called_once_with(
            '/path/to/output.nc',
            'africa.nc',
            'application/x-netcdf4',
            location='s3://example-bucket/',
            logger=variable_subsetter.logger
        )

    def test_asynchronous_request(self,
                                  mock_stage,
                                  mock_subset_granule,
                                  mock_get_mimetype):
        """ A request that specified `isSynchronous = False` should complete
            for a single granule. It should call the `subset_granule` function,
            and then indicate the request completed.

        """
        mock_subset_granule.return_value = '/path/to/output.nc'
        mock_get_mimetype.return_value = ('application/x-netcdf4', None)

        message = self.create_message('C1233860183-EEDTEST',
                                      'G1233860471-EEDTEST',
                                      ['/home/tests/data/africa.nc'],
                                      ['alpha_var', 'blue_var'],
                                      'ealdrin',
                                      False)

        variable_subsetter = HarmonyAdapter(message)
        with patch.object(HarmonyAdapter, 'process_item', self.process_item_spy):
            variable_subsetter.invoke()
        granule = variable_subsetter.message.granules[0]

        mock_subset_granule.assert_called_once_with(granule.url,
                                                    granule.variables,
                                                    ANY,
                                                    variable_subsetter.logger)
        mock_get_mimetype.assert_called_once_with('/path/to/output.nc')

        mock_stage.assert_called_once_with(
            '/path/to/output.nc',
            'africa.nc',
            'application/x-netcdf4',
            location='s3://example-bucket/',
            logger=variable_subsetter.logger)

    def test_unspecified_synchronous_request(self,
                                             mock_stage,
                                             mock_subset_granule,
                                             mock_get_mimetype):
        """ A request the does not specify `isSynchronous` should default to
            synchronous behaviour. The `subset_granule` function should be
            called. Then the request should complete.

        """
        mock_subset_granule.return_value = '/path/to/output.nc'
        mock_get_mimetype.return_value = ('application/x-netcdf4', None)

        message = self.create_message('C1233860183-EEDTEST',
                                      'G1233860471-EEDTEST',
                                      ['/home/tests/data/africa.nc'],
                                      ['alpha_var', 'blue_var'],
                                      'mcollins')

        variable_subsetter = HarmonyAdapter(message)
        with patch.object(HarmonyAdapter, 'process_item', self.process_item_spy):
            variable_subsetter.invoke()
        granule = variable_subsetter.message.granules[0]

        mock_subset_granule.assert_called_once_with(granule.url,
                                                    granule.variables,
                                                    ANY,
                                                    variable_subsetter.logger)
        mock_get_mimetype.assert_called_once_with('/path/to/output.nc')

        mock_stage.assert_called_once_with(
            '/path/to/output.nc',
            'africa.nc',
            'application/x-netcdf4',
            location='s3://example-bucket/',
            logger=variable_subsetter.logger
        )

    def test_missing_granules(self,
                              mock_stage,
                              mock_subset_granule,
                              mock_get_mimetype):
        """ A request with no specified granules in an inbound Harmony message
            should raise an exception.

        """
        mock_subset_granule.return_value = '/path/to/output.nc'
        mock_get_mimetype.return_value = ('application/x-netcdf4', None)

        message = self.create_message('C1233860183-EEDTEST',
                                      'G1233860471-EEDTEST',
                                      [],
                                      ['alpha_var', 'blue_var'],
                                      'pconrad',
                                      False)

        variable_subsetter = HarmonyAdapter(message)
        error = None
        try:
            with patch.object(HarmonyAdapter, 'process_item', self.process_item_spy):
                variable_subsetter.invoke()
        except Exception as e:
            error = e

        mock_subset_granule.assert_not_called()
        mock_get_mimetype.assert_not_called()

        mock_stage.assert_not_called()
        assert str(error) == 'No granules specified for variable subsetting'

    def test_synchronous_multiple_granules(self,
                                           mock_stage,
                                           mock_subset_granule,
                                           mock_get_mimetype):
        """ A request for synchronous processing, with multiple granules
            specified should raise an exception.

        """
        output_paths = ['/path/to/output1.nc', '/path/to/output2.nc']

        mock_subset_granule.side_effect = output_paths
        mock_get_mimetype.return_value = ('application/x-netcdf4', None)

        message = self.create_message('C1233860183-EEDTEST',
                                      'G1233860471-EEDTEST',
                                      ['/home/tests/data/africa.nc',
                                       '/home/tests/data/VNL2_test.nc'],
                                      ['alpha_var', 'blue_var'],
                                      'rgordon',
                                      True)

        variable_subsetter = HarmonyAdapter(message)
        error = None
        try:
            with patch.object(HarmonyAdapter, 'process_item', self.process_item_spy):
                variable_subsetter.invoke()
        except Exception as e:
            error = e

        mock_subset_granule.assert_not_called()
        mock_get_mimetype.assert_not_called()

        mock_stage.assert_not_called()
        assert str(error) == 'Synchronous requests accept only one granule'

    def test_asynchronous_multiple_granules(self,
                                            mock_stage,
                                            mock_subset_granule,
                                            mock_get_mimetype):
        """ A request for asynchronous processing, with multiple granules
            specified should be successful, and call `subset_granule` for each
            input granule.

        """
        output_paths = ['/path/to/output1.nc', '/path/to/output2.nc']
        output_filenames = ['africa.nc', 'VNL2_test.nc']

        mock_subset_granule.side_effect = output_paths
        mock_get_mimetype.return_value = ('application/x-netcdf4', None)

        message = self.create_message('C1233860183-EEDTEST',
                                      'G1233860471-EEDTEST',
                                      ['/home/tests/data/africa.nc',
                                       '/home/tests/data/VNL2_test.nc'],
                                      ['alpha_var', 'blue_var'], 'abean',
                                      False)

        variable_subsetter = HarmonyAdapter(message)
        with patch.object(HarmonyAdapter, 'process_item', self.process_item_spy):
            variable_subsetter.invoke()
        granules = variable_subsetter.message.granules

        print(mock_stage.mock_calls)

        for index, granule in enumerate(granules):
            mock_subset_granule.assert_any_call(granule.url,
                                                granule.variables,
                                                ANY,
                                                variable_subsetter.logger)
            mock_get_mimetype.assert_any_call(output_paths[index])

            mock_stage.assert_any_call(
                output_paths[index],
                output_filenames[index],
                'application/x-netcdf4',
                location=message.stagingLocation,
                logger=variable_subsetter.logger
            )

    def test_missing_variables(self,
                               mock_stage,
                               mock_subset_granule,
                               mock_get_mimetype):
        """ Ensure that if no variables are specified for a source, the service
            raises an exception.

        """
        mock_subset_granule.return_value = '/path/to/output.nc'
        mock_get_mimetype.return_value = ('application/x-netcdf4', None)

        message = self.create_message('C1233860183-EEDTEST',
                                      'G1233860471-EEDTEST',
                                      ['/home/tests/data/africa.nc'],
                                      [],
                                      'jlovell')

        variable_subsetter = HarmonyAdapter(message)
        error = None
        try:
            with patch.object(HarmonyAdapter, 'process_item', self.process_item_spy):
                variable_subsetter.invoke()
        except Exception as e:
            error = e

        mock_subset_granule.assert_not_called()
        mock_get_mimetype.assert_not_called()

        mock_stage.assert_not_called()
        assert str(error) == 'No variables specified for subsetting'
