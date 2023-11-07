FROM python:3.8.16-slim-bullseye

ENV USER=user \
    UID=1000 \
    HOME=/home/user
RUN adduser \
        --disabled-password \
        --gecos "Default user" \
        --uid ${UID} \
        --home ${HOME} \
        --force-badname \
        ${USER}

COPY requirements.txt ./

RUN apt-get update -y && apt-get install -y \
    && pip3 install --no-cache-dir --no-deps -r requirements.txt \
    && rm -rf /usr/local/src/*

ENV PYTHONPATH=${HOME}

WORKDIR ${HOME}
USER ${NB_USER}

# see src/backend/setup.txt to download the models
COPY models ./models
COPY data ./data
COPY src ./src
