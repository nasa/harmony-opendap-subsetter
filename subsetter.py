""" Data Services variable subsetter service for Harmony.

This service will take an input file

Message schema:

{
    "format": {},
    "granules": [{"variables": ["alpha_var"],
                  "url": "/home/tests/data/africa.nc"}],
    "isSynchronous": true,
    "user": "urs_username"
}

"isSynchronous" can be set to either true or false, or omitted, in which case
the service will behave synchronously.

"""
from argparse import ArgumentParser
from logging import Logger
import os

import harmony

from pymods.subset import subset_granule
from pymods.utilities import get_granule_mimetype


class HarmonyAdapter(harmony.BaseHarmonyAdapter):
    """ This class extends the BaseHarmonyAdapter class, to implement the
        `invoke` method, which performs variable subsetting.

        Note: Harmony currently only supports multiple files for asynchronous
        requests. For synchronous requests this service will only handle the
        first granule in the message. If the Harmony message doesn't specify
        "isSynchronous", the default behaviour is to assume the request is
        synchronous.

    """

    def invoke(self):
        """ Callback used by BaseHarmonyAdapter to invoke the service. """
        self.logger.info('Starting Data Services variable subsetter service')
        os.environ['HDF5_DISABLE_VERSION_CHECK'] = '1'


        operations = {'is_variable_subset': True,
                      'is_regridded': False,
                      'is_subsetted': False}

        try:
            self.validate_message()

            for index, granule in enumerate(self.message.granules):
                self.logger.info(f'Downloading: {granule.url}')
                self.download_granules([granule])

                output_file_path = subset_granule(granule, self.logger)

                if not self.message.isSynchronous:
                    # TODO: Should mimetype be based on output file?
                    mimetype = get_granule_mimetype(granule)

                    # This progress assumes granules are similar sizes.
                    progress = int((100 * (index + 1)) /
                                   len(self.message.granules))

                    self.async_add_local_file_partial_result(
                        output_file_path, source_granule=granule,
                        mime=mimetype[0], progress=progress, title=granule.id,
                        **operations
                    )

            if self.message.isSynchronous:
                # TODO: Should mimetype be based on output file?
                mimetype = get_granule_mimetype(self.message.granules[0])

                self.completed_with_local_file(
                    output_file_path,
                    source_granule=self.message.granules[0],
                    mime=mimetype[0],
                    **operations
                )
            else:
                self.async_completed_successfully()

            self.logger.info('Variable subsetting completed.')

        except Exception as error:
            self.logger.info('Variable subsetting failed:')
            self.logger.exception(error)
            self.completed_with_error('Variable subsetting failed with error: '
                                      f'{str(error)}')

        finally:
            self.cleanup()

    def validate_message(self):
        """ Check the service was triggered by a valid message containing
            the expected number of granules.

        """
        if not hasattr(self, 'message'):
            raise Exception('No message request')

        if self.message.isSynchronous is None:
            # Set default behaviour to synchronous
            self.message.isSynchronous = True

        if self.message.isSynchronous:
            self.logger.info('Service has been called synchronously.')
        else:
            self.logger.info('Service has been called asynchronously.')

        if not hasattr(self.message, 'granules') or not self.message.granules:
            raise Exception('No granules specified for variable subsetting')
        elif self.message.isSynchronous and len(self.message.granules) > 1:
            # TODO: remove this condition when synchronous requests can handle
            # multiple granules.
            raise Exception('Synchronous requests accept only one granule')


if __name__ == '__main__':
    """ Enable this command to be run locally within a conda environment
        containing all the requisite packages specified in both the
        conda_requirements.txt and pip_requirements.txt files.

    """
    PARSER = ArgumentParser(
        prog='Variable Subsetting',
        description='Run the Data Services variable subsetting Tool'
    )
    PARSER.add_argument(
        '--harmony-action',
        choices=['invoke'],
        help='The action Harmony needs to perform (currently only "invoke")'
    )
    PARSER.add_argument(
        '--harmony-input',
        help='The input data for the action provided by Harmony'
    )

    ARGS = PARSER.parse_args()
    harmony.run_cli(PARSER, ARGS, HarmonyAdapter)
