#
# Service image for ghcr.io/nasa/harmony-opendap-subsetter, a Harmony backend
# service that includes variable, temporal and spatial subsetting.
# Operations are performed via requests to an instance of OPeNDAP in the Cloud
# to retrieve the requested variables from an Earth Observation scientific
# file.
#
# This image installs dependencies via pip into the system Python of the
# official Python base image. The service code is then copied into the Docker
# image.
#
# 2020-05-07: Initial version inspired by Swath Projector.
# 2021-05-17: Copy Pip requirements file after conda environment creation.
# 2022-02-07: Updated conda environment to Python 3.8.
# 2023-10-02: Updated conda environment to Python 3.11.
# 2023-10-06: Updated pymods directory to HOSS, conda environment name to hoss.
# 2026-09-21: Replaced conda with the official Python base image. All
#             dependencies, including the HDF5 and netCDF-C libraries bundled
#             in the netCDF4 wheel, are installed via Pip.
#
FROM python:3.12-slim-trixie

WORKDIR "/home"

# Copy pip dependencies into the container
COPY ./pip_requirements.txt pip_requirements.txt

# Install pip dependencies
RUN pip install --no-input --no-cache-dir --root-user-action=ignore -r pip_requirements.txt

# Bundle app source
COPY ./hoss hoss

# Configure a container to be executable via the `docker run` command.
ENTRYPOINT ["python", "-m", "hoss"]
