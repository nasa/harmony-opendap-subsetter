#
# Using Conda within ENTRYPOINT was taken from:
# https://pythonspeed.com/articles/activate-conda-dockerfile/
#

FROM continuumio/miniconda3

WORKDIR "/home"

# Bundle app source
COPY ./pymods pymods
COPY ./subsetter.py subsetter.py
COPY ./conda_requirements.txt conda_requirements.txt
COPY ./pip_requirements.txt pip_requirements.txt

# Create Conda environment
RUN conda create --name subsetter --file conda_requirements.txt python=3.7 \
	--channel conda-forge \
	--channel defaults

# Make RUN commands use the Conda environment
SHELL ["conda", "run", "--name", "subsetter", "/bin/bash", "-c"]

# Install additional Pip dependencies
RUN pip install -r pip_requirements.txt

ENTRYPOINT ["conda", "run", "--name", "subsetter", "PYTHONPATH=.", "python", "subsetter.py"]
