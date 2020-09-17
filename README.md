## Data Services service for variable subsetting

This repository contains the Data Services variable subsetter offered as a
[Harmony](https://wiki.earthdata.nasa.gov/spaces/viewspace.action?key=HARMONY)
service. It will be capable of receiving a granule, with a list of variables to
return in an output file. This output file will contain both those variables
specified in the input message, and the relevant supporting variables (such as
coordinates).

Currently, if the request is synchronous, the variable subsetter will only
allow processing of a single granule, and will raise an exception if there are
multiple granules defined.

### Local usage:

To download source code:

```bash
git clone https://git.earthdata.nasa.gov/scm/sitc/var_subsetter.git var_subsetter
```

To build the Docker image:

```bash
cd var_subsetter
./bin/build-image
```

To run the service in a Docker container, after building the image:

```bash
./bin/run-image \
	--harmony-action 'invoke' \
	--harmony-input '{"sources": [{"variables": [{"id": "V0001-EXAMPLE", "name": "science_variable", "fullPath": "/path/to/science_variable"}], "granules": [{"url": "..."}]}], "user": "urs_user", "isSynchronous": true, "callback": "URL for callback"}'
```

The `--harmony-action` and `--harmony-input` parameters mimic how Harmony would
invoke the variable subsetter service. A set of example messages can be found
[here](https://git.earthdata.nasa.gov/projects/HARMONY/repos/harmony-service-lib-py/browse/tests/example_messages.py). In the example above, the message requests
for a subset of a granule (at URL: "...") to contain the "/path/to/science"
variable. The variable subsetter will extract this and other variables that it
refers to in its CF attributes.

Passing environment variables to the Docker container is done via a `.env` file
in the root of the repository. The `bin/run-image` will automatically detect
the presence of this file. To run the variable subsetter service container
locally, use the following template for `.env`. This template assumes there is
no Harmony local instance running.

```
EDL_USERNAME=jglenn
EDL_PASSWORD=Fr13ndsh1p7
ENV=test
```

Note, when `ENV` is set to `test` or `dev`, the Harmony adapter will not try to
stage the output results unless a local Harmony instance is present. However, a
`callback` is still required in the Harmony message, as the logging will try to
concatenate this string in a warning stating that the service will not reply to
Harmony.

### Development notes:

The variable subsetter runs within a Docker container (both the project itself,
and the tests that are run for CI/CD. If you add a new Python package to be
used within the project (or remove a third party package), the change in
dependencies will need to be recorded in the relevant requirements file:

* `var_subsetter/conda_requirements.txt`: Requirements needed for the subsetter
	to run, obtained from the `conda-forge` channel.
* `var_subsetter/pip_requirements.txt`: Additional requirements installed
	within the container's conda environment via Pip. These are also required
	for the source code of the variable subsetter to run.
* `var_subsetter/test/pip_test_requirements.txt`: Requirements only used while
	running tests, such as `pylint` or `coverage`. These are kept separate to
	reduce the dependencies in the delivered software.

### Running tests:

The variable subsetter has Python tests that use the `unittest` package. These
can be run within a Docker container using the following two scripts:

```bash
./bin/build-test
./bin/run-test /full/path/to/var_subsetter-coverage
```

Coverage reports are being generate for each build in Bamboo, and saved as artifacts.
Following URL is an example coverage report in Bamboo:

https://ci.earthdata.nasa.gov/artifact/HITC-SVS20/BRT/build-3/Coverage-Report/source/tests/coverage/index.html

For faster iteration, one can use the `build-env` and `run-dev` scripts:

```bash
./bin/build-env

# Do some development
./bin/run-dev
# Do some more development
./bin/run-dev
```

The `build-env` script creates a conda environment installing the requisite
Python packages from conda and pip. The `run-dev` script uses the output from
`build-env` as a base image, and copies the source code for the variable
subsetter into a new container, which is then executed to run the `unittest`
suite.
