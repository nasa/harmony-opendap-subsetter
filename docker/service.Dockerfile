#
# Service image for sds/variable-subsetter, a Harmony backend service that
# includes both a variable subsetter and the Harmony OPeNDAP Subsetter (HOSS).
# Both perform requests to an instance of OPeNDAP in the Cloud to retrieve
# the requested variables from an Earth Observation scientific file, with HOSS
# also supporting bounding box spatial subsetting for geographically gridded
# data.
#
# This image instantiates a conda environment, with required pacakges, before
# installing additional dependencies via Pip. The service code is then copied
# into the Docker image, before environment variables are set to activate the
# created conda environment.
#
# Updated: 2021-06-24
#
FROM continuumio/miniconda3

WORKDIR "/home"

# Copy Conda requirements into the container
COPY ./conda_requirements.txt conda_requirements.txt

# Create Conda environment
RUN conda create -y --name subsetter --file conda_requirements.txt python=3.8 -q \
	--channel conda-forge \
	--channel defaults

# Copy additional Pip dependencies into the container
COPY ./pip_requirements.txt pip_requirements.txt

# Install additional Pip dependencies
RUN conda run --name subsetter pip install --no-input -r pip_requirements.txt

# Bundle app source
COPY ./pymods pymods
COPY ./subsetter.py subsetter.py

# Set conda environment to subsetter, as conda run will not stream logging.
# Setting these environment variables is the equivalent of `conda activate`.
ENV _CE_CONDA='' \
    _CE_M='' \
    CONDA_DEFAULT_ENV=subsetter \
    CONDA_EXE=/opt/conda/bin/conda \
    CONDA_PREFIX=/opt/conda/envs/subsetter \
    CONDA_PREFIX_1=/opt/conda \
    CONDA_PROMPT_MODIFIER=(subsetter) \
    CONDA_PYTHON_EXE=/opt/conda/bin/python \
    CONDA_ROOT=/opt/conda \
    CONDA_SHLVL=2 \
    PATH="/opt/conda/envs/subsetter/bin:${PATH}" \
    SHLVL=1

# Configure a container to be executable via the `docker run` command.
ENTRYPOINT ["python", "subsetter.py"]
