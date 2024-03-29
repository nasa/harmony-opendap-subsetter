# Listing of test files

* ABoVE_TVPRM_bbox.nc4
  - This is an example output from a bounding box request to the ABoVE
    TVPRM collection, where -160 ≤ longitude (degrees east) ≤ -145, 68 ≤ latitude
    (degrees north) ≤ 70. Note, ABoVE TVPRM has 8784 time slices. To minimise the
    size of the stored artefact in Git, this example output only contains the first
    10 time slices. ABoVE TVPRM has a projected grid that uses an Albers Conical
    Equal Area CRS.
* ABoVE_TVPRM_example.dmr
  - An example `.dmr` file for the ABoVE TVPRM collection, as obtained from OPeNDAP
* ABoVE_TVPRM_prefetch.nc4
  - An example dimension prefetch output from OPeNDAP for the ABoVE TVPRM collection.
    This contains the `time`, `x` and `y` variables
* ATL03_example.dmr
  - An example `.dmr` file from the ICESat-2/ATL03 collection, as obtained from
    OPeNDAP. ATL03 is a trajectory data set and should only be used (currently)
    with the variable subsetting operations of the service
* GPM_3IMERGHH_bounds.nc4
  - An example output from a bounding box and temporal subset request for
    a collection containing bounds variables. The bounding box is -30 ≤ longitude
    (degrees east) ≤ -15, 45 ≤ latitude (degrees north) ≤ 60
* GPM_3IMERGHH_example.dmr
  - An example `.dmr` file for the GPM/3IMERGHH collection, as obtained from
    OPeNDAP. GPM/3IMERGHH has a half-hourly time dimension to the grid. It also
    contains bounds variable references
* GPM_3IMERGHH_prefetch.nc4
  - An example dimension prefetch output from OPeNDAP for the GPM/IMERGHH
    collection. This contains the `/Grid/time`, `/Grid/lat`, and `/Grid/lon`
    dimension variables along with their associated bounds variables
* M2T1NXSLV_example.dmr
  - An example `.dmr` file for a MERRA-2 collection. Granules in this collection
    are geographically gridded and contain a time dimension that has half-hour
    resolution
* M2T1NXSLV_geo_temporal.nc4
  - Example output for a MERRA-2 collection. This output is for both a spatial
    (bounding box) and temporal subset
* M2T1NXSLV_prefetch.nc4
  - An example dimension prefetch output from OPeNDAP for a MERRA-2 collection.
    This contains a longitude, latitude and temporal dimension
* M2T1NXSLV_temporal.nc4
  - An example output for a MERRA-2 collection, with a request to OPeNDAP for
    a temporal subset
* f16_ssmis_20200102v7.nc
  - An input granule for the RSSMIF16D collection. The variables in this
    collection a 3-dimensional, gridded with a latitude and longitude dimension
    and a single element time dimension. RSSMIF16D also has 0 ≤ longitude
    (degrees east) ≤ 360
* f16_ssmis_filled.nc
  - The output from a spatial subset request when the requested bounding box
    crosses the Prime Meridian (and therefore the edge of the grid)
* f16_ssmis_geo.nc
  - The output from a spatial and variable subset request with a bounding box
* f16_ssmis_geo_desc.nc
  - The output from a spatial and variable subset request with a bounding box
    input. The latitude dimension is also descending in this example, unlike
    the native ascending ordering
* f16_ssmis_geo_no_vars.nc
  - The results of a spatial subset only with a bounding box
* f16_ssmis_lat_lon.nc
  - The output for a prefetch request that retrieves only the latitude and
    longitude variables for the RSSMIF16D collection. This sample predates
    the temporal subsetting work, and so does not also include the temporal
    dimension variable
* f16_ssmis_lat_lon_desc.nc
  - The output for a prefetch request that retrieves only the latitude and
    longitude variables for the RSSMIF16D collection. This differs from the
    previous sample file, as the latitude dimension is descending in this file
* f16_ssmis_unfilled.nc
  - The sample output for a request to OPeNDAP when the longitude range crosses
    the edge of the bounding box. In this case, the data retrieved from OPeNDAP
    will be a band in latitude, but cover the full longitudinal range. HOSS
    will later fill the required region of that band before returning the result
    to the end-user
* rssmif16d_example.dmr
  - An example `.dmr` file as retrieved from OPeNDAP for the RSSMIF16D collection
* ATL16_prefetch.dmr
  - An example `.dmr` file retrieved from OPeNDAP for the ATL16 collection, but whittled
    down to only contain the six required dimension variables.
* ATL16_prefetch.nc4
  - A sample output file that contains the six required dimension variables and one
    requested science variable in the ATL16 collection.
* ATL16_prefetch_group.dmr
  - An example `.dmr` file that is identical to the `ATL16_prefetch.dmr` file
    except for an additional fabricated nested group variable, whereas all the variables in the
    ATL16 collection are in the root directory.
* ATL16_prefetch_group.nc4
  - A sample output file that is nearly identical to the `ATL16_prefetch.nc4` file except
    for an additional fabricated nested group variable (the same one in the
    ATL16_prefetch_group.dmr file).
* ATL16_prefetch_bnds.dmr
  - An example `.dmr` file that is nearly identical to the `ATL16_prefetch.dmr` file
    except for four additional fabricated variables that represented the four
    possible cases of combining bounds variable existence and cell alignment.
