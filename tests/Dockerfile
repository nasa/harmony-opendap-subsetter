FROM continuumio/miniconda3

ARG EDL_USERNAME
ARG EDL_PASSWORD
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR "/home"

# Bundle app source
COPY ./tests tests
COPY ./pymods pymods
COPY subsetter.py .
COPY conda_requirements.txt .
COPY pip_requirements.txt .

# Create Conda environment
RUN conda create --name subsetter --file conda_requirements.txt python=3.7 \
	--channel conda-forge \
	--channel defaults

# Make RUN commands use the Conda environment
SHELL ["conda", "run", "--name", "subsetter", "/bin/bash", "-c"]

# Install additional Pip requirements (for the variable subsetter itself)
RUN pip install -r pip_requirements.txt

# Install additional Pip requirements (for testing)
RUN pip install -r tests/pip_test_requirements.txt

# Install Harmony
# Note GIT credentials could have issue if password contains certain characters.
RUN pip install "git+https://${EDL_USERNAME}:${EDL_PASSWORD}@git.earthdata.nasa.gov/scm/harmony/harmony-service-lib-py.git"

# Configure a container to be executable via the `docker run` command.
ENTRYPOINT ["conda", "run", "--name", "subsetter", "/home/tests/run_tests.sh"]
