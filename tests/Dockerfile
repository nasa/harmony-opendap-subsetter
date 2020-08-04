FROM continuumio/miniconda3

ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR "/home"

# Bundle app source
COPY ./tests tests
COPY ./pymods pymods
COPY subsetter.py .
COPY conda_requirements.txt .
COPY pip_requirements.txt .

# Create Conda environment
RUN conda create --name subsetter --file conda_requirements.txt python=3.7 -q \
	--channel conda-forge \
	--channel defaults

# Install additional Pip requirements (for the variable subsetter itself)
RUN conda run --name subsetter pip install -r pip_requirements.txt

# Install additional Pip requirements (for testing)
RUN conda run --name subsetter pip install -r tests/pip_test_requirements.txt

# Set conda environment to subsetter, as conda run will not stream logging.
# Setting these environment variables is the equivalent of `conda activate`.
ENV _CE_CONDA ''
ENV _CE_M ''
ENV CONDA_DEFAULT_ENV subsetter
ENV CONDA_EXE /opt/conda/bin/conda
ENV CONDA_PREFIX /opt/conda/envs/subsetter
ENV CONDA_PREFIX_1 /opt/conda
ENV CONDA_PROMPT_MODIFIER (subsetter)
ENV CONDA_PYTHON_EXE /opt/conda/bin/python
ENV CONDA_ROOT /opt/conda
ENV CONDA_SHLVL 2
ENV PATH "/opt/conda/envs/subsetter/bin:${PATH}"
ENV SHLVL 1

# Configure a container to be executable via the `docker run` command.
ENTRYPOINT ["/home/tests/run_tests.sh"]
