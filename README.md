# The Harmony OPeNDAP SubSetter (HOSS)   

This repository contains source code and CI/CD for a
[Harmony](https://wiki.earthdata.nasa.gov/spaces/viewspace.action?key=HARMONY)
backend service. The Harmony OPeNDAP SubSetter (HOSS) performs variable,
spatial, temporal and named-dimension subsetting by making requests to
[OPeNDAP](https://www.opendap.org/support/user-documentation).

The service is published as a Docker image, from which Harmony creates
Kubernetes pods upon deployment. To ensure this occurs, HOSS is defined in the
main [services.yml](https://github.com/nasa/harmony/blob/main/config/services.yml)
configuration file for Harmony, and with different UMM-S records.

HOSS require data to be ingested such that the granules can be accessed via
OPeNDAP in the Cloud (Hyrax). This means:

* Generating a sidecar `.dmrpp` file. The file should then be placed in the
  same location (S3 bucket) as the granule itself, and have the same filename,
  with an additional `.dmrpp` suffix.
* Having a `RelatedUrl` entry in the UMM-G record for each granule with a `Type`
  of `USE SERVICE API` and `Subtype` of `OPENDAP DATA`. This can be generated
  via [a Cumulus task](https://github.com/nasa/cumulus/tree/master/tasks/hyrax-metadata-updates).

## HOSS Capabilities:

The Harmony OPeNDAP SubSetter (HOSS) is designed for use with gridded data
(levels 3 or 4). HOSS can perform spatial, temporal and variable subsetting.

To perform a successful subset with HOSS, a collection must:

* Be configured for data access via OPeNDAP. This means having a `.dmrpp`
  sidecar file with each granule, and having the OPeNDAP URL within the
  related URLs of each UMM-G record.
* Contain 1-D dimension variables within each granule.
* Each gridded variable must refer to the 1-D dimension variables for the grid.
  Failing this, configuration file entries must be provided to override missing
  CF-Convention metadata.

Variable subsets will include both variables requested by the end-user, as well
as variables that are referred to in CF-Convention attributes of the requested
variables. Such references include coordinates and grid mappings.

Spatial and temporal subsets are accomplished by identifying grid dimension
variables, which are retrieved in full. The requested spatial or temporal
ranges are then converted to indices corresponding to the dimension pixel
containing that value. The spatial and temporal subsets are then retrieved by
including those indices for all gridded variables in the OPeNDAP DAP4
constraint expression.

To perform a successful temporal subset, it is expected that the temporal
variables adhere to [CF-Conventions](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.9/cf-conventions.html#time-coordinate).

Given that HOSS retrieves data from OPeNDAP using ranges of indices, the
returned output for each variable is a hyperrectangle. This precludes shape
file spatial subsetting via only HOSS. Instead, when receiving a shape file in
an inbound Harmony message, HOSS determines a bounding box that minimally
encompasses all features in the GeoJSON shape file. HOSS will then use this
bounding box to perform a spatial subset, before sending its output to
MaskFill. MaskFill will then assign fill values to all pixels within the HOSS
hyperrectangle, but outside the GeoJSON shape.

Note, if both a bounding box and a shape file are specified in the same Harmony
message, HOSS will use the bounding box information from the message.

### Dimensions that can be subsetted:

* Dimensions must be monotonic, e.g., the array must either always increase or
  always decrease as the values are scanned from start to finish.
* If a dimension does not have an associated `bounds` array, the values stored
  in the dimension array must be in the centre points of the pixels.
* If a dimension has an associated `bounds` array, as indicated by the `bounds`
  CF-Convention metadata attribute, these bounds will be used to determine
  index ranges. Such dimensions with `bounds` arrays can therefore have
  dimension pixel values anywhere within the pixel, not just the centre.

### Variable Subsetter:

An additional service chain is specified within the Harmony `services.yml`,
which utilises the HOSS Docker image, but only allows users to perform variable
subsetting. This is designed for non-gridded collections with data available
via OPeNDAP.

As with HOSS, an end-user can specify the variables they want, and the Variable
Subsetter will also retrieve any variables referred to in specific
CF-Convention metadata attributes of the requested variables. This ensures the
output files from the Variable Subsetter remain viable as relevant information,
such as spatial and temporal coordinates, cannot be omitted from the output due
to being forgotten in the initial request to Harmony.

## Contributing:

Contributions are welcome! For more information, see `CONTRIBUTING.md`.

## Developing:

Development within this repository should occur on a feature branch. Pull
Requests (PRs) are created with a target of the `main` branch before being
reviewed and merged.

Releases are created when a feature branch is merged to `main` and that branch
also contains an update to the `docker/service_version.txt` file.

### Local usage:

To download source code:

```bash
git clone https://github.com/nasa/harmony-opendap-subsetter
```

To build the Docker image:

```bash
cd harmony-opendap-subsetter
./bin/build-image
```

HOSS is best run locally using a local instance of Harmony, available from
[here](https://github.com/nasa/harmony). After building the service Docker
image locally via the `./bin/build-image` script, make sure
your local Harmony instance lists "hoss" in the comma-separated list of
services under the `LOCALLY_DEPLOYED_SERVICES` environment variable contained
in your local Harmony `.env` file.

Once a local Harmony instance is configured to run HOSS, the service can be
invoked to run on UAT-hosted data via
[`harmony-py`](https://github.com/nasa/harmony-py):

```
from harmony import BBox, Client, Collection, Environment, Request

harmony_client = Client(env=Environment.LOCAL)
bbox = BBox(w=20, s=-10, e=40, n=10)
collection = Collection(id='<UAT collection concept ID>')
variables = ['/fullpath/var_one', '/fullpath/var_two']
request = Request(collection=collection, variables=variables, spatial=bbox,
				  max_results=1)

job_id = harmony_client.submit(request)
```

Requests should be visible at `localhost:3000/jobs`. Debugging can be performed
by looking at logs for the Kubernetes pods associated with Harmony in Docker
Desktop.

Note: There are issues if you try to have a Python environment (e.g., conda)
that contains both the `harmony-py` and `harmony-service-lib-py` packages.

### Development notes:

HOSS runs within a Docker container (both the project itself, and the tests
that are run for CI/CD. If you add a new Python package to be used within the
project (or remove a third party package), the change in dependencies will need
to be recorded in the relevant requirements file:

* `conda_requirements.txt`: Requirements needed for HOSS to run, obtained from
  the `conda-forge` channel.
* `pip_requirements.txt`: Additional requirements installed within the
  container's conda environment via Pip. These are also required for the source
  code of HOSS to run. It is preferable to obtain dependencies from Pip, so
  that Pip can manage inter-package dependencies and version conflicts.
* `tests/pip_test_requirements.txt`: Requirements only used while running
  tests, such as `pylint` or `coverage`. These are kept separate to reduce the
  dependencies in the delivered software.

### Running tests:

HOSS has Python tests that use the `unittest` package. These can be run within
a Docker container using the following scripts:

```bash
# Build the service image, which is a base for the test image
./bin/build-image

# Build the test image.
./bin/build-test

# Run the tests.
./bin/run-test
```

Coverage reports are being generate for each build in Bamboo, and saved as
artefacts.

### Versioning

As a Harmony service, HOSS follows semantic version numbers (e.g.,
`major.minor.patch`). This version is included in the
`docker/service_version.txt` file. When updating the Python service code, the
version number contained in the `service_version.txt` file should be
incremented before creating a pull request.

In addition, the changes for the new service version should be described at a
high level in `CHANGELOG.md`.

The general rules for which version number to increment are:

* Major: When API changes are made to the service that are not backwards
  compatible.
* Minor: When functionality is added in a backwards compatible way.
* Patch: Used for backwards compatible bug fixes or performance improvements,
  these changes should not affect how an end user calls the service.

When the Docker image is built as part of the publication GitHub workflow, it
will be tagged with the semantic version number as stored in
`docker/service_version.txt`.

The semantic version number of the service was set to 1.0.0 when migrating to
GitHub.

## CI/CD:

The CI/CD for HOSS is contained in GitHub workflows in the `.github/workflows`
directory:

* `run_tests.yml` - A reusable workflow that builds the service and test Docker
  images, then runs the Python unit test suite in an instance of the test
  Docker container.
* `run_tests_on_pull_requests.yml` - Triggered for all PRs against the `main`
  branch. It runs the workflow in `run_tests.yml` to ensure all tests pass for
  the new code.
* `publish_docker_image.yml` - Triggered either manually or for commits to the
  `main` branch that contain changes to the `docker/service_version.txt` file.

The `publish_docker_image.yml` workflow will:

* Run the full unit test suite, to prevent publication of broken code.
* Extract the semantic version number from `docker/service_version.txt`.
* Extract the release notes for the most recent version from `CHANGELOG.md`
* Build the service Docker image and push it to the GitHub Container Registry.
* Create a GitHub release that will also tag the related git commit with the
  semantic version number.

Before triggering a release, ensure both the `docker/service_version.txt` and
`CHANGELOG.md` files are updated. The `CHANGELOG.md` file requires a specific
format for a new release, as it looks for the following string to define the
newest release of the code (starting at the top of the file).

```
## vX.Y.Z
```

### pre-commit hooks:

This repository uses [pre-commit](https://pre-commit.com/) to enable pre-commit
checking the repository for some coding standard best practices. These include:

* Removing trailing whitespaces.
* Removing blank lines at the end of a file.
* JSON files have valid formats.
* [ruff](https://github.com/astral-sh/ruff) Python linting checks.
* [black](https://black.readthedocs.io/en/stable/index.html) Python code
  formatting checks.

To enable these checks:

```bash
# Install pre-commit Python package as part of test requirements:
pip install -r tests/pip_test_requirements.txt

# Install the git hook scripts:
pre-commit install

# (Optional) Run against all files:
pre-commit run --all-files
```

When you try to make a new commit locally, `pre-commit` will automatically run.
If any of the hooks detect non-compliance (e.g., trailing whitespace), that
hook will state it failed, and also try to fix the issue. You will need to
review and `git add` the changes before you can make a commit.

It is planned to implement additional hooks, possibly including tools such as
`mypy`.

## Get in touch:

You can reach out to the maintainers of this repository via email:

* david.p.auty@nasa.gov
* owen.m.littlejohns@nasa.gov
