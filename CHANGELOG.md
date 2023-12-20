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
