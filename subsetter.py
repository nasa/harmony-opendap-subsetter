""" Data Services variable subsetter service for Harmony.

This service will take an input file

Message schema:

{
    "sources": [
        {
            "variables": [
                {
                    "fullPath": "/path/to/science_variable",
                    "id": "V0001-EXAMPLE",
                    "name": "science_variable"
                },
            ],
            "granules": [
                {
                    "url": "/home/tests/data/africa.nc"
                }
            ]
        }
    ],
    "callback": "URL for callback",
    "isSynchronous": true,
    "user": "urs_username"
}

"isSynchronous" can be set to either true or false, or omitted, in which case
the service will behave synchronously.

"""
from argparse import ArgumentParser
import shutil
from tempfile import mkdtemp
from pystac import Asset

import harmony
from harmony.util import generate_output_filename

from pymods.subset import subset_granule
from pymods.utilities import get_file_mimetype


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
        """
        Adds validation to default process_item-based invocation

        Returns
        -------
        pystac.Catalog
            the output catalog
        """
        self.validate_message()
        return super().invoke()

    def process_item(self, item, source):
        """
        Processes a single input item.  Services that are not aggregating multiple input files
        should prefer to implement this method rather than #invoke

        This example copies its input to the output, marking "dpi" and "variables" message
        attributes as having been processed

        Parameters
        ----------
        item : pystac.Item
            the item that should be processed
        source : harmony.message.Source
            the input source defining the variables, if any, to subset from the item

        Returns
        -------
        pystac.Item
            a STAC catalog whose metadata and assets describe the service output
        """
        result = item.clone()
        result.assets = {}

        # Create a temporary dir for processing we may do
        workdir = mkdtemp()
        try:
            # Get the data file
            asset = None
            for k, v in item.assets.items():
                if v.roles and 'opendap' in v.roles:
                    asset = v
                    break
                elif v.roles and 'data' in v.roles:
                    # Legacy workflows won't provide a data role of 'opendap'.
                    # After workflows are converted to chaining, this can all be
                    # condensed to:
                    # asset = next(v for k, v in item.assets.items() if 'opendap' in (v.roles or []))
                    asset = v

            # Mark any fields the service processes so later services do not repeat work
            variables = source.process('variables')  # Variable subsetting

            # Subset
            output_file_path = subset_granule(asset.href, variables, workdir, self.logger)

            # Stage the output file with a conventional filename
            mime, _ = get_file_mimetype(output_file_path)
            staged_filename = generate_output_filename(asset.href, variable_subset=source.variables)
            url = harmony.util.stage(output_file_path,
                                     staged_filename,
                                     mime,
                                     location=self.message.stagingLocation,
                                     logger=self.logger)

            # Update the STAC record
            result.assets['data'] = Asset(url, title=staged_filename, media_type=mime, roles=['data'])

            # Return the STAC record
            return result
        finally:
            # Clean up any intermediate resources
            shutil.rmtree(workdir)

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

        has_granules = hasattr(self.message, 'granules') and self.message.granules
        has_items = False
        try:
            has_items = bool(self.catalog and next(self.catalog.get_all_items()))
        except StopIteration:
            pass

        if not has_granules and not has_items:
            raise Exception('No granules specified for variable subsetting')
        elif self.message.isSynchronous and len(self.message.granules) > 1:
            # TODO: remove this condition when synchronous requests can handle
            # multiple granules.
            raise Exception('Synchronous requests accept only one granule')

        # TODO: Update this check on variables once their format has expanded.
        for source in self.message.sources:
            if not hasattr(source, 'variables') or not source.variables:
                raise Exception('No variables specified for subsetting')


if __name__ == '__main__':
    # Enable this command to be run locally within a conda environment
    # containing all the requisite packages specified in both the
    # conda_requirements.txt and pip_requirements.txt files.
    PARSER = ArgumentParser(
        prog='Variable Subsetting',
        description='Run the Data Services variable subsetting Tool'
    )
    harmony.setup_cli(PARSER)
    ARGS, _ = PARSER.parse_known_args()
    harmony.run_cli(PARSER, ARGS, HarmonyAdapter)
