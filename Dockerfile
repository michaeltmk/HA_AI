FROM pytorch/pytorch:latest

WORKDIR /usr/src/

# Set Environment Variable
ENV LC_ALL=C.UTF-8

RUN apt update && gpt upgrade
RUN set -x; apt-get install -y --no-install-recommends unzip
RUN apt install python3 python3-pip

# Install dependencies
ADD REQUIREMENT.txt /opt/requirements.txt
RUN python3 -m pip install --upgrade pip && pip3 install -r /opt/requirements.txt

# Add main script
ADD main.py /usr/src/main.py

# Unzip model file
ADD efficientnetb0_v1_first_trial_4* /usr/src/
RUN unzip /usr/src/efficientnetb0_v1_first_trial_4.zip

# Add required dir
RUN mkdir -p /enigma/datasets/HA-Sample/
RUN mkdir -p /enigma/local_storage/result/
COPY . .
VOLUME ["/enigma", "/usr/src"]
