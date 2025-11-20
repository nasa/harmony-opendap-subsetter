"""Harmony OPeNDAP SubSetter (HOSS).

This service uses the `HarmonyBaseAdapter.process_item` method to subset a
single image at a time. Requests for multiple granules will invoke the
service once per image, performing the subset on each granule in separate
calls to HOSS.

An invocation of HOSS provides several sources of information:

* `pystac.Item` - direct input to the `process_item` method. This contains
  `pystac.Asset` objects for the specific granule being processed.
* `harmony.message.Source` - direct input to the `process_item` method.
  This contains information on the collection to which the granule being
  processed belongs.
* `harmony.message.Message` - input when instantiating the `HossAdapter`.
  This contains the subset request information, such as bounding box,
  GeoJSON shape file path, temporal range or variables list where needed.
* `harmony.util.Config` - input when instantiating the `HossAdapter`. This
  `namedtuple` contains necessary configuration information, such as OAUTH
  information for the Harmony EDL application being used, and AWS staging
  location information.
* `pystac.Catalog` - input when instantiating the `HossAdapter`. This
  contains all requested granules, and is iterated through with individual
  calls to `process_item` for each granule.

"""

import shutil
from tempfile import mkdtemp

from harmony_service_lib import BaseHarmonyAdapter
from harmony_service_lib.message import Source
from harmony_service_lib.util import HarmonyException, generate_output_filename, stage
from pystac import Asset, Item

from hoss.dimension_utilities import is_index_subset
from hoss.harmony_log_context import set_logger
from hoss.subset import subset_granule
from hoss.utilities import get_file_mimetype, raise_from_hoss_exception


class HossAdapter(BaseHarmonyAdapter):
    """This class extends the BaseHarmonyAdapter class, to implement the
    `invoke` method, which performs variable, spatial and temporal
    subsetting via requests to OPeNDAP.

    """

    def __init__(self, message, catalog=None, config=None):
        """Initialize the HossAdapter"""
        super().__init__(message, catalog=catalog, config=config)
        set_logger(self.logger)

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

    def process_item(self, item: Item, source: Source):
        """Processes a single input item. Services that are not aggregating
        multiple input files should prefer to implement this method rather
        than `invoke`

        This example copies its input to the output, marking `variables`
        and `subset.bbox` message attributes as having been processed

        Parameters
        ----------
        item : pystac.Item
            the item that should be processed
        source : harmony.message.Source
            the input source defining the variables, if any, to subset from
            the item

        Returns
        -------
        pystac.Item
            a STAC catalog whose metadata and assets describe the service
            output

        """
        result = item.clone()
        result.assets = {}

        # Create a temporary dir for processing we may do
        workdir = mkdtemp()
        try:
            # Get the data file
            asset = next(
                (
                    item_asset
                    for item_asset in item.assets.values()
                    if 'opendap' in (item_asset.roles or [])
                ),
                None,
            )

            self.logger.info(f'Collection short name: {source.shortName}')

            # Invoke service logic to retrieve subset of file from OPeNDAP
            output_file_path = subset_granule(
                asset.href, source, workdir, self.message, self.config
            )

            # Stage the output file with a conventional filename
            mime, _ = get_file_mimetype(output_file_path)
            staged_filename = generate_output_filename(
                asset.href,
                variable_subset=source.variables,
                ext='.nc4',
                is_subsetted=(
                    is_index_subset(self.message) or len(source.variables) > 0
                ),
            )
            url = stage(
                output_file_path,
                staged_filename,
                mime,
                location=self.message.stagingLocation,
                logger=self.logger,
            )

            # Update the STAC record
            result.assets['data'] = Asset(
                url, title=staged_filename, media_type=mime, roles=['data']
            )

        except Exception as exception:
            self.logger.exception(exception)
            raise_from_hoss_exception(exception)
        finally:
            # Clean up any intermediate resources
            shutil.rmtree(workdir)

        # Return the STAC record
        return result

    def validate_message(self):
        """Check the service was triggered by a valid message containing
        the expected number of granules.

        """
        if not hasattr(self, 'message'):
            raise HarmonyException('No message request')

        if self.message.isSynchronous is None:
            # Set default behaviour to synchronous
            self.message.isSynchronous = True

        if self.message.isSynchronous:
            self.logger.info('Service has been called synchronously.')
        else:
            self.logger.info('Service has been called asynchronously.')

        has_granules = hasattr(self.message, 'granules') and self.message.granules

        try:
            has_items = bool(self.catalog and next(self.catalog.get_all_items()))
        except StopIteration:
            has_items = False

        if not has_granules and not has_items:
            raise HarmonyException('No granules specified for variable subsetting')

        for source in self.message.sources:
            if not hasattr(source, 'variables') or not source.variables:
                self.logger.info('All variables will be retrieved.')
