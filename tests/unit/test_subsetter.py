from typing import List, Optional
from unittest import TestCase
from unittest.mock import patch
import json
import os

from harmony import BaseHarmonyAdapter
from harmony.message import Message

from subsetter import HarmonyAdapter
from tests.utilities import contains


@patch('subsetter.get_file_mimetype')
@patch('subsetter.subset_granule')
@patch.object(BaseHarmonyAdapter, 'cleanup')
@patch.object(BaseHarmonyAdapter, 'async_completed_successfully')
@patch.object(BaseHarmonyAdapter, 'async_add_local_file_partial_result')
@patch.object(BaseHarmonyAdapter, 'completed_with_error')
@patch.object(BaseHarmonyAdapter, 'completed_with_local_file')
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

    def create_message(self, collection: str, granule_id: str, file_paths: List[str],
                       variable_list: List[str], user: str,
                       is_synchronous: Optional[bool] = None) -> Message:
        """ Create a Harmony Message object with the requested attributes. """
        granules = [{'id': granule_id, 'url': file_path}
                    for file_path in file_paths]
        variables = [{'name': variable} for variable in variable_list]
        message_content = {'sources': [{'collection': collection,
                                        'granules': granules,
                                        'variables': variables}],
                           'user': user}

        if is_synchronous is not None:
            message_content['isSynchronous'] = is_synchronous

        return Message(json.dumps(message_content))

    def test_synchronous_request(self, mock_completed_with_local_file,
                                 mock_completed_with_error,
                                 mock_async_add_local_file_partial,
                                 mock_async_completed, mock_cleanup,
                                 mock_subset_granule, mock_get_mimetype):
        """ A request that specifies `isSynchronous = True` should complete
            for a single granule. It should call the `subset_granule` function,
            and then indicate the request completed.

        """
        mock_subset_granule.return_value = '/path/to/output.nc'
        mock_get_mimetype.return_value = ('application/octet-stream', None)

        message = self.create_message('C1233860183-EEDTEST',
                                      'G1233860471-EEDTEST',
                                      ['/home/tests/data/africa.nc'],
                                      ['alpha_var', 'blue_var'],
                                      'narmstrong',
                                      True)
        variable_subsetter = HarmonyAdapter(message)
        variable_subsetter.invoke()
        granule = variable_subsetter.message.granules[0]

        mock_subset_granule.assert_called_once_with(granule,
                                                    variable_subsetter.logger)
        mock_get_mimetype.assert_called_once_with('/path/to/output.nc')

        mock_completed_with_local_file.assert_called_once_with(
            '/path/to/output.nc', source_granule=granule,
            mime='application/octet-stream', **self.operations
        )
        mock_async_add_local_file_partial.assert_not_called()
        mock_async_completed.assert_not_called()
        mock_cleanup.assert_called_once()
        mock_completed_with_error.assert_not_called()

    def test_asynchronous_request(self, mock_completed_with_local_file,
                                  mock_completed_with_error,
                                  mock_async_add_local_file_partial,
                                  mock_async_completed, mock_cleanup,
                                  mock_subset_granule, mock_get_mimetype):
        """ A request that specified `isSynchronous = False` should complete
            for a single granule. It should call the `subset_granule` function,
            and then indicate the request completed.

        """
        mock_subset_granule.return_value = '/path/to/output.nc'
        mock_get_mimetype.return_value = ('application/octet-stream', None)

        message = self.create_message('C1233860183-EEDTEST',
                                      'G1233860471-EEDTEST',
                                      ['/home/tests/data/africa.nc'],
                                      ['alpha_var', 'blue_var'],
                                      'ealdrin',
                                      False)

        variable_subsetter = HarmonyAdapter(message)
        variable_subsetter.invoke()
        granule = variable_subsetter.message.granules[0]

        mock_subset_granule.assert_called_once_with(granule,
                                                    variable_subsetter.logger)
        mock_get_mimetype.assert_called_once_with('/path/to/output.nc')

        mock_completed_with_local_file.assert_not_called()
        mock_async_add_local_file_partial.assert_called_once_with(
            '/path/to/output.nc', source_granule=granule, progress=100,
            mime='application/octet-stream', title=granule.id,
            **self.operations
        )
        mock_async_completed.assert_called_once()
        mock_cleanup.assert_called_once()
        mock_completed_with_error.assert_not_called()

    def test_unspecified_synchronous_request(self,
                                             mock_completed_with_local_file,
                                             mock_completed_with_error,
                                             mock_async_add_local_file_partial,
                                             mock_async_completed, mock_cleanup,
                                             mock_subset_granule,
                                             mock_get_mimetype):
        """ A request the does not specify `isSynchronous` should default to
            synchronous behaviour. The `subset_granule` function should be
            called. Then the request should complete.

        """
        mock_subset_granule.return_value = '/path/to/output.nc'
        mock_get_mimetype.return_value = ('application/octet-stream', None)

        message = self.create_message('C1233860183-EEDTEST',
                                      'G1233860471-EEDTEST',
                                      ['/home/tests/data/africa.nc'],
                                      ['alpha_var', 'blue_var'],
                                      'mcollins')

        variable_subsetter = HarmonyAdapter(message)
        variable_subsetter.invoke()
        granule = variable_subsetter.message.granules[0]

        mock_subset_granule.assert_called_once_with(granule,
                                                    variable_subsetter.logger)
        mock_get_mimetype.assert_called_once_with('/path/to/output.nc')

        mock_completed_with_local_file.assert_called_once_with(
            '/path/to/output.nc',
            source_granule=granule,
            mime='application/octet-stream',
            **self.operations
        )
        mock_async_add_local_file_partial.assert_not_called()
        mock_async_completed.assert_not_called()
        mock_cleanup.assert_called_once()
        mock_completed_with_error.assert_not_called()

    def test_missing_granules(self, mock_completed_with_local_file,
                              mock_completed_with_error,
                              mock_async_add_local_file_partial,
                              mock_async_completed, mock_cleanup,
                              mock_subset_granule, mock_get_mimetype):
        """ A request with no specified granules in an inbound Harmony message
            should raise an exception.

        """
        mock_subset_granule.return_value = '/path/to/output.nc'
        mock_get_mimetype.return_value = ('application/octet-stream', None)

        message = self.create_message('C1233860183-EEDTEST',
                                      'G1233860471-EEDTEST',
                                      [],
                                      ['alpha_var', 'blue_var'],
                                      'pconrad')

        variable_subsetter = HarmonyAdapter(message)
        variable_subsetter.invoke()

        mock_subset_granule.assert_not_called()
        mock_get_mimetype.assert_not_called()

        mock_completed_with_local_file.assert_not_called()
        mock_async_add_local_file_partial.assert_not_called()
        mock_async_completed.assert_not_called()
        mock_cleanup.assert_called_once()
        mock_completed_with_error.assert_called_with(
            contains('No granules specified for variable subsetting')
        )

    def test_synchronous_multiple_granules(self,
                                           mock_completed_with_local_file,
                                           mock_completed_with_error,
                                           mock_async_add_local_file_partial,
                                           mock_async_completed, mock_cleanup,
                                           mock_subset_granule,
                                           mock_get_mimetype):
        """ A request for synchronous processing, with multiple granules
            specified should raise an exception.

        """
        output_paths = ['/path/to/output1.nc', '/path/to/output2.nc']

        mock_subset_granule.side_effect = output_paths
        mock_get_mimetype.return_value = ('application/octet-stream', None)

        message = self.create_message('C1233860183-EEDTEST',
                                      'G1233860471-EEDTEST',
                                      ['/home/tests/data/africa.nc',
                                       '/home/tests/data/VNL2_test.nc'],
                                      ['alpha_var', 'blue_var'],
                                      'rgordon',
                                      True)

        variable_subsetter = HarmonyAdapter(message)
        variable_subsetter.invoke()

        mock_subset_granule.assert_not_called()
        mock_get_mimetype.assert_not_called()

        mock_completed_with_local_file.assert_not_called()
        mock_async_add_local_file_partial.assert_not_called()
        mock_async_completed.assert_not_called()
        mock_cleanup.assert_called_once()
        mock_completed_with_error.assert_called_with(
            contains('Synchronous requests accept only one granule')
        )

    def test_asynchronous_multiple_granules(self,
                                            mock_completed_with_local_file,
                                            mock_completed_with_error,
                                            mock_async_add_local_file_partial,
                                            mock_async_completed, mock_cleanup,
                                            mock_subset_granule,
                                            mock_get_mimetype):
        """ A request for asynchronous processing, with multiple granules
            specified should be successful, and call `subset_granule` for each
            input granule.

        """
        output_paths = ['/path/to/output1.nc', '/path/to/output2.nc']
        progresses = [50, 100]

        mock_subset_granule.side_effect = output_paths
        mock_get_mimetype.return_value = ('application/octet-stream', None)

        message = self.create_message('C1233860183-EEDTEST',
                                      'G1233860471-EEDTEST',
                                      ['/home/tests/data/africa.nc',
                                       '/home/tests/data/VNL2_test.nc'],
                                      ['alpha_var', 'blue_var'], 'abean',
                                      False)

        variable_subsetter = HarmonyAdapter(message)
        variable_subsetter.invoke()
        granules = variable_subsetter.message.granules

        mock_completed_with_local_file.assert_not_called()
        self.assertEqual(mock_async_add_local_file_partial.call_count,
                         len(granules))
        mock_async_completed.assert_called_once()
        mock_cleanup.assert_called_once()
        mock_completed_with_error.assert_not_called()

        for index, granule in enumerate(granules):
            mock_subset_granule.assert_any_call(granule,
                                                variable_subsetter.logger)
            mock_get_mimetype.assert_any_call(output_paths[index])
            mock_async_add_local_file_partial.assert_any_call(
                output_paths[index], source_granule=granule,
                progress=progresses[index], mime='application/octet-stream',
                title=granule.id,
                **self.operations
            )

    def test_missing_variables(self, mock_completed_with_local_file,
                               mock_completed_with_error,
                               mock_async_add_local_file_partial,
                               mock_async_completed, mock_cleanup,
                               mock_subset_granule, mock_get_mimetype):
        """ Ensure that if no variables are specified for a source, the service
            raises an exception.

        """
        mock_subset_granule.return_value = '/path/to/output.nc'
        mock_get_mimetype.return_value = ('application/octet-stream', None)

        message = self.create_message('C1233860183-EEDTEST',
                                      'G1233860471-EEDTEST',
                                      ['/home/tests/data/africa.nc'],
                                      [],
                                      'jlovell')

        variable_subsetter = HarmonyAdapter(message)
        variable_subsetter.invoke()

        mock_subset_granule.assert_not_called()
        mock_get_mimetype.assert_not_called()

        mock_completed_with_local_file.assert_not_called()
        mock_async_add_local_file_partial.assert_not_called()
        mock_async_completed.assert_not_called()
        mock_cleanup.assert_called_once()
        mock_completed_with_error.assert_called_with(
            contains('No variables specified for subsetting')
        )
