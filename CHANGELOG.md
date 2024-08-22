
## v1.0.5
### 2024-08-19

This version of HOSS updates the `is_index_subset` method to check for empty list (in case of dimensions subset)
as well as None (for the spatial, bbox and temporal subsets). This prevents 1-D dimension variables from being
unnecessarily requested from OPeNDAP for variable subsets which needs to be done only for spatial and temporal
subsetting requests. This also prevents a whole granule request when 1-D dimension variables were not present
in the granule.

## v1.0.4
### 2024-04-05

This version of HOSS implements `black` code formatting across the repository.
There should be no functional changes in the service.

## v1.0.3
### 2024-03-29

This version of HOSS handles the error in the crs_wkt attribute in ATL19 where the
north polar crs variable has a leading iquotation mark escaped by back slash in the
crs_wkt attribute. This causes errors when the projection is being interpreted from
the crs variable attributes.

## v1.0.2
### 2024-02-26

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

## v1.0.1
### 2023-12-19

This version of HOSS removes custom, redundant download retry logic. Instead
retries are relied upon from `harmony-service-lib` and for each stage in a
Harmony workflow.

## v1.0.0
### 2023-10-06

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
