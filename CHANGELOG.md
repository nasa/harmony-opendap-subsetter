## [v1.1.17] - 2025-11-25

### Changed

- Change HOSS behavior to return No Data Warning instead of a failed exception
  when the variable, spatial, temporal request does not return any data.

## [v1.1.16] - 2025-11-17

### Fixed

- Add tolerance when comparing grid extent float values in the function
  `remove_points_outside_grid_extents` to prevent inaccurate spatial subset
  results. Some issues caused by DAS-2424 updates are being resolved in this
  release as part of DAS-2456.

## [v1.1.15] - 2025-10-24

### Changed

- HOSS errors that will not succeed with an unchanged retry, are raised to the
  harmony service as `NoRetryException`s to prevent wasted CPU cycles in
  Harmony.

- HOSS logging is updated to use `set_logger` and `get_logger` from a new
  `harmony_log_context` module. Rather than pass the adapter's logger to all of
  the functions, any function may now call `get_logger()` to retrieve it.

## [v1.1.14] - 2025-10-22

### Fixed

- Adds support for temporal requests with only start or end time specified.
  When a Harmony message omits the start time, HOSS now defaults to
  `0001-01-01T00:00:00.000Z`. When the end time is omitted, HOSS defaults to
  the current time. This allows temporal subsetting to work with
  partially-specified time ranges.  These changes do not affect requests
  without a temporal component or fully qualified temporal requests.

## [v1.1.13] - 2025-10-21

### Changed

- Updates evaluation of bbox or polygon constraint to exclude areas outside the
  projected target grid. An error exception occurs if spatial constraint is
  entirely outside the projected grid extents.
- Updates tests to be less dependent on architecture when comparing floats.
- Fixes test that modified a source file fixture.
- Changes infrastructure so that local and Docker runs of the tests produce
  output in same locations.
- GitHub once again captures the artifacts from the tests and coverage.


## [v1.1.12] - 2025-10-15

### Added

- Adds SMAP string variables as excluded variables list in the
`earthdata-varinfo` configuration file. An exception is thrown when an excluded
variable is explicitly requested.

## [v1.1.11] - 2025-10-15

### Changed

- The `harmony-service-lib` dependency has been updated to v2.8.1 to make use of
  increased timeouts, mitigating timeout issues from OPeNDAP beginning to send
  the first byte to HOSS.
- Release notes for HOSS will now include the commit history for that release.

## [v1.1.10] - 2025-08-21

This version of HOSS updates the conda environment to support Python 3.12. This also updates the
dependent packages to their latest supportable versions and updates Harmony Service Library version.

## [v1.1.9] - 2025-08-13

This version of HOSS updates fixes invalid output extents in cases where the
requested bounding area (bbox, shape) extends beyond valid locations for the native
projection of the data. E.g., EASE-GRID-2 North Polar (LAEA) projection becomes INF
at latitude -90°.

## [v1.1.8] - 2025-05-23

This version of HOSS updates the hoss configuration file to include
Ancillary_Data group in SPL3FTA collection metadata overrides for dimension
variables in that group.

## [v1.1.7] - 2025-05-21

This version of HOSS updates the hoss configuration file to include
EASE_column_index and EASE_row_index as ancillary variables for the SPL2SMAP_S
collection.

## [v1.1.6] - 2025-02-24

This version of HOSS adds support for multiple coordinate variables that
are in the same group for cases where coordinates are used in place of
dimensions. This also supports exclusive requests of coordinate
datasets that do not have coordinate attributes or dimensions.

## [v1.1.5] - 2025-02-14

This version of HOSS adds support for 3D variables which
do not have the nominal order. This would provide support
for the 3D variables in SMAP - SPL3SMP with dimension order
information provided in the configurations file.

## [v1.1.4] - 2025-02-12

This version of HOSS adds support for SMAP L3 polar variables that are unable to have their
dimension scale arrays created from their corresponding lat/lon variables. A 'master geotransform'
attribute has been added to the grid mapping reference variable for the affected collections
and function updates were made to create the dimension arrays from the master geotransform
when it is present.

## [v1.1.3] - 2025-01-29

This version of HOSS supports configuration updates to hoss_config.json to
add dimension configurations for 3D variables. Functions were updated to provide the
ability to spatial subset 3D variables for products like SMAP L3 which did
not have dimension arrays.

## [v1.1.2] - 2025-01-20

- [[DAS-2256](https://bugs.earthdata.nasa.gov/browse/DAS-2256)]
  HOSS has been updated to use `earthdata-varinfo` version 3.0.1.
  Please see the
  [earthdata-varinfo release notes](https://github.com/nasa/earthdata-varinfo/releases/tag/3.0.1)
  for more information.

## [v1.1.1] - 2025-01-14

This version of HOSS merges the feature branch that contains V1.1.0 to the main branch.
Additional updates included code quality improvements with additional unit tests, revised methodology
in functions that selected the data points from the coordinate datasets and calculation of the dimension
arrays. Functions were added to determine the dimension order for 2D variables. These updates,
V1.1.1 and V1.1.0 are entirely to support SMAP L3 data - in particular SPL2SMAP_S, SPL3SMAP, SPL3SMA -
all of which have “anonymous” dimensions (without dimension names and dimension variables).  No functional
updates are present, nor changes that effect existing collection support.

## [v1.1.0] - 2024-11-25

This version of HOSS provides support for gridded products that do not contain
CF-Convention compliant grid mapping variables and 1-D dimension variables, such
as SMAP L3. Functions updated to retrieve coordinate attributes and grid mappings,
using overrides specified in the hoss_config.json configuration file.
This implementation uses the latitude/longitude values across one row, and one
column to project it to the target grid using Proj and to compute the the X-Y
dimension arrays. Functions have been added to check any fill values present in
the coordinate variable data. Check for the dimension order for 2D datasets is
done using the latitude and longitude data varying across the row versus the
column. Support for multiple grids is handled by associating the group-name into
the cache-name for coordinates already processed for dimension ranges.
Several new functions related to this implementation have been added to
a new module `coordinate_utilities.py`.

## [v1.0.5] - 2024-08-19

This version of HOSS updates the `is_index_subset` method to check for empty list (in case of dimensions subset)
as well as None (for the spatial, bbox and temporal subsets). This prevents 1-D dimension variables from being
unnecessarily requested from OPeNDAP for variable subsets which needs to be done only for spatial and temporal
subsetting requests. This also prevents a whole granule request when 1-D dimension variables were not present
in the granule.

## [v1.0.4] - 2024-04-05

This version of HOSS implements `black` code formatting across the repository.
There should be no functional changes in the service.

## [v1.0.3] - 2024-03-29

This version of HOSS handles the error in the crs_wkt attribute in ATL19 where the
north polar crs variable has a leading iquotation mark escaped by back slash in the
crs_wkt attribute. This causes errors when the projection is being interpreted from
the crs variable attributes.

## [v1.0.2] - 2024-02-26

This version of HOSS correctly handles edge-aligned geographic collections by
adding the attribute `cell_alignment` with the value `edge` to `hoss_config.json`
for edge-aligned collections (namely, ATL16), and by adding functions that
create pseudo bounds for edge-aligned collections to make HOSS use the
`dimension_utilities.py` function, `get_dimension_indices_from_bounds`. The
pseudo bounds are only used internally and are not returned in the HOSS subset.

This change also includes an addition of a CF override that addresses an
issue with the ATL16 metadata for the variables `/spolar_asr_obs_grid` and
`/spolar_lorate_blowing_snow_freq` where their `grid_mapping` attribute points
to north polar variables instead of south polar variables. This CF Override
can be removed if/when the metadata is corrected.

## [v1.0.1] - 2023-12-19

This version of HOSS removes custom, redundant download retry logic. Instead
retries are relied upon from `harmony-service-lib` and for each stage in a
Harmony workflow.

## [v1.0.0] - 2023-10-06

This version of the Harmony OPeNDAP SubSetter (HOSS) contains all functionality
previously released internally to EOSDIS as `sds/variable-subsetter:0.3.6`.
Minor reformatting of the repository structure has occurred to comply with
recommended best practices for a Harmony backend service repository, but the
service itself is functionally unchanged. Additional contents to the repository
include updated documentation and files outlined by the
[NASA open-source guidelines](https://code.nasa.gov/#/guide).

Repository structure changes include:

* Migrating `pymods` directory to `hoss`.
* Migrating `subsetter.py` to `hoss/adapter.py`.
* Addition of `hoss/main.py`.

For more information on internal releases prior to NASA open-source approval,
see legacy-CHANGELOG.md.
[v1.1.17]: https://github.com/nasa/harmony-opendap-subsetter/releases/tag/1.1.17
[v1.1.16]: https://github.com/nasa/harmony-opendap-subsetter/releases/tag/1.1.16
[v1.1.15]: https://github.com/nasa/harmony-opendap-subsetter/releases/tag/1.1.15
[v1.1.14]: https://github.com/nasa/harmony-opendap-subsetter/releases/tag/1.1.14
[v1.1.13]: https://github.com/nasa/harmony-opendap-subsetter/releases/tag/1.1.13
[v1.1.12]: https://github.com/nasa/harmony-opendap-subsetter/releases/tag/1.1.12
[v1.1.11]: https://github.com/nasa/harmony-opendap-subsetter/releases/tag/1.1.11
[v1.1.10]: https://github.com/nasa/harmony-opendap-subsetter/releases/tag/1.1.10
[v1.1.9]: https://github.com/nasa/harmony-opendap-subsetter/releases/tag/1.1.9
[v1.1.8]: https://github.com/nasa/harmony-opendap-subsetter/releases/tag/1.1.8
[v1.1.7]: https://github.com/nasa/harmony-opendap-subsetter/releases/tag/1.1.7
[v1.1.6]: https://github.com/nasa/harmony-opendap-subsetter/releases/tag/1.1.6
[v1.1.5]: https://github.com/nasa/harmony-opendap-subsetter/releases/tag/1.1.5
[v1.1.4]: https://github.com/nasa/harmony-opendap-subsetter/releases/tag/1.1.4
[v1.1.3]: https://github.com/nasa/harmony-opendap-subsetter/releases/tag/1.1.3
[v1.1.2]: https://github.com/nasa/harmony-opendap-subsetter/releases/tag/1.1.2
[v1.1.1]: https://github.com/nasa/harmony-opendap-subsetter/releases/tag/1.1.1
[v1.1.0]: https://github.com/nasa/harmony-opendap-subsetter/releases/tag/1.1.0
[v1.0.5]: https://github.com/nasa/harmony-opendap-subsetter/releases/tag/1.0.5
[v1.0.4]: https://github.com/nasa/harmony-opendap-subsetter/releases/tag/1.0.4
[v1.0.3]: https://github.com/nasa/harmony-opendap-subsetter/releases/tag/1.0.3
[v1.0.2]: https://github.com/nasa/harmony-opendap-subsetter/releases/tag/1.0.2
[v1.0.1]: https://github.com/nasa/harmony-opendap-subsetter/releases/tag/1.0.1
[v1.0.0]: https://github.com/nasa/harmony-opendap-subsetter/releases/tag/1.0.0
