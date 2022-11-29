## v0.3.4
### 2022-11-23

This version of HOSS is primarily internal code clean-up, in particular making
more extensive use of the Harmony `Message` object throughout the service.

## v0.3.3
### 2022-10-26

This version of HOSS updates the configuration file settings to also support
SMAP level 4 collections. In addition, the `sds-varinfo` dependency is updated
to version 3.0.0, and so the configuration file used by `VarInfoFromDmr` has
been converted to a JSON format.

## v0.3.2
### 2022-09-23

This version of HOSS updates the service to use bounds variables to determine
dimension index ranges when they are present in a granule. (e.g, `/lon_bnds`).
If bounds variables are not present, the dimension values themselves will be
used as before.

In addition to this change, a bug fix is included in
`pymods.dimension_utilities.get_dimension_index_range` to ensure requested
dimension ranges including a value of 0 are handled correctly.

## v0.3.1
### 2022-09-07

This version of HOSS updates the service to extract the collection short name
from the input Harmony message and STAC item. This collection short name is
then used by `sds-varinfo` when instantiating the `VarInfoFromDmr` object used
to map variable and dimension dependencies. This change supports collections
where the short name is not stored as a global attribute directly within a
granule.

## v0.3.0
### 2022-08-25

This version of HOSS updates the service to support spatial subsetting of
collections with projected grids (e.g., non-geographic). This functionality
works by utilising the chained service of HOSS and MaskFill. HOSS derives an
approximate minimum geographic resolution of the projected grid and populates
points around the perimeter of the input bounding box or GeoJSON polygon(s) at
this resolution. The minimum and maximum indices of the points corresponding to
these perimeter points is determined. The rectangular region in the projected
horizontal spatial plane is retrieved from OPeNDAP as before. The output is
then sent to MaskFill in order to fill those points that are within the
retrieved rectangular portion of the array, but outside of the specified
GeoJSON shape.

## v0.2.2
### 2022-07-11

This version of HOSS updates the `harmony-service-lib` dependency to v1.0.20
in support of changes to the handling of STAC objects.

## v0.2.1
### 2022-07-07

This version of HOSS updates the error handling around requests made to OPeNDAP
to leverage the new `ServerException` in the
[harmony-service-lib Python package](https://github.com/nasa/harmony-service-lib-py).

## v0.2.0
### 2022-07-07

This version of HOSS expands the service capabilities to support general
dimension subsetting as Harmony moves to support generic dimension subsetting
via the API (e.g., `subset=generalDimensionName(min:max)`). Existing
functionality is used to derive index ranges. If a dimension is specifically
named, but also specified by a bounding box or temporal range in the Harmony
message, the latter takes precedence (e.g., `Message.subset.bbox`,
`Message.subset.shape` and `Message.temporal` take priority over
`Message.subset.dimensions`).

## v0.1.4
### 2022-06-01

This version of HOSS adds a bug fix that ensures all-variable requests for
temporal and shape file spatial subsetting will succeed. This is done by
ensuring that temporal and shape file spatial subsetting are correctly
identified as requiring index-range subsets to be specified in the OPeNDAP DAP4
constraint expression.

## 0.1.3
### 2022-05-23

This version of HOSS explicitly updates the version of the `sds-varinfo`
dependency to 1.2.5. This ensures HOSS better handles variables that do not
have a `units` metadata attribute when checking if a variable is temporal.

## 0.1.2
### 2022-05-23

This version of HOSS adds the capability to derive a minimally encompassing
bounding box from an input GeoJSON shape file, as specified via
`Message.subset.shape` within the Harmony message. This minimally encompassing
bounding box will then be used to derive index ranges for variables requested
from OPeNDAP. This functionality supports a chained service with MaskFill, such
that the two Harmony services can be combined to provide a shape file spatial
subset.

## 0.0.2
### 2022-02-09

This version of HOSS implements temporal subsetting within the service. The
temporal information is derived from the `Message.temporal` parameter of an
input Harmony message, with a single temporal range being supported.

## 0.0.1
### 2022-01-05

This version of HOSS adopts semantic version numbering using
`docker/service_version.txt` to ensure Docker images published for the HOSS
services can be easily tracked and selected by Harmony. At this point, HOSS
offers bounding box spatial subsetting and variable subsetting for
geographically gridded L3/L4 collections.
