FROM continuumio/miniconda3

ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR "/home"

# Install dependencies first. These steps take the longest and change the least,
# so their layers should usually be cached
# Run bin/build-test latest --no-delete to take advantage of the faster build

# Copy tests and requirements files
COPY tests/pip_test_requirements.txt .
COPY conda_requirements.txt .
COPY pip_requirements.txt .

# Create Conda environment
RUN conda create -y --name subsetter --file conda_requirements.txt python=3.7 -q \
	--channel conda-forge \
	--channel defaults

# Install additional Pip requirements (for the variable subsetter itself)
COPY pip_requirements.txt .
RUN conda run --name subsetter pip install -r pip_requirements.txt

# Install additional Pip requirements (for testing)
RUN conda run --name subsetter pip install --no-input -r pip_test_requirements.txt

# Bundle app source
COPY ./tests tests
COPY ./pymods pymods
COPY subsetter.py .

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
ENTRYPOINT ["/home/tests/run_tests.sh"]
