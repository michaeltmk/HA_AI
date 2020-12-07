FROM pytorch/pytorch:latest

WORKDIR /usr/src/
USER root

# Set Environment Variable
ENV LC_ALL=C.UTF-8

RUN apt-get update
RUN set -x; apt-get install -y --no-install-recommends p7zip-full python3 python3-pip

# Install dependencies
ADD REQUIREMENT.txt /opt/requirements.txt
RUN python3 -m pip install --upgrade pip && pip3 install -r /opt/requirements.txt

# Add main script
ADD main.py /usr/src/main.py
ADD main.py ~/submission/main.py

# Unzip model file
RUN mkdir /opt/model
ADD efficientnetb0_v1_first_trial_4* /opt/model/
RUN cd /usr/src && 7z x /opt/model/efficientnetb0_v1_first_trial_4.zip
RUN rm -rf /opt/model

# Add required dir
RUN mkdir -p /enigma/datasets/HA-Sample/
RUN mkdir -p /enigma/local_storage/result/
COPY . .
VOLUME ["/enigma", "/usr/src"]

COPY ./entrypoint.sh /
ENTRYPOINT ["/entrypoint.sh"]
