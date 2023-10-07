#
# Service image for ghcr.io/nasa/harmony-opendap-subsetter, a Harmony backend
# service that includes variable, temporal and spatial subsetting.
# Operations are performed via requests to an instance of OPeNDAP in the Cloud
# to retrieve the requested variables from an Earth Observation scientific
# file.
#
# This image instantiates a conda environment, with required pacakges, before
# installing additional dependencies via Pip. The service code is then copied
# into the Docker image, before environment variables are set to activate the
# created conda environment.
#
# 2020-05-07: Initial version inspired by Swath Projector.
# 2021-05-17: Copy Pip requirements file after conda environment creation.
# 2022-02-07: Updated conda environment to Python 3.8.
# 2023-10-02: Updated conda environment to Python 3.11.
# 2023-10-06: Updated pymods directory to HOSS, conda environment name to hoss.
#
FROM continuumio/miniconda3

WORKDIR "/home"

# Copy Conda requirements into the container
COPY ./conda_requirements.txt conda_requirements.txt

# Create Conda environment
RUN conda create -y --name hoss --file conda_requirements.txt python=3.11 -q \
	--channel conda-forge \
	--channel defaults

# Copy additional Pip dependencies into the container
COPY ./pip_requirements.txt pip_requirements.txt

# Install additional Pip dependencies
RUN conda run --name hoss pip install --no-input -r pip_requirements.txt

# Bundle app source
COPY ./hoss hoss

# Set conda environment to "hoss", as conda run will not stream logging.
# Setting these environment variables is the equivalent of `conda activate`.
ENV _CE_CONDA='' \
    _CE_M='' \
    CONDA_DEFAULT_ENV=hoss \
    CONDA_EXE=/opt/conda/bin/conda \
    CONDA_PREFIX=/opt/conda/envs/hoss \
    CONDA_PREFIX_1=/opt/conda \
    CONDA_PROMPT_MODIFIER=(hoss) \
    CONDA_PYTHON_EXE=/opt/conda/bin/python \
    CONDA_ROOT=/opt/conda \
    CONDA_SHLVL=2 \
    PATH="/opt/conda/envs/hoss/bin:${PATH}" \
    SHLVL=1

# Configure a container to be executable via the `docker run` command.
ENTRYPOINT ["python", "-m", "hoss"]
