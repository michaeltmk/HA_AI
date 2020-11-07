FROM pytorch/pytorch:latest

WORKDIR /usr/src/

# Set Environment Variable
ENV LC_ALL=C.UTF-8

RUN apt-get update
RUN set -x; apt-get install -y --no-install-recommends unzip python3 python3-pip

# Install dependencies
ADD REQUIREMENT.txt /opt/requirements.txt
RUN python3 -m pip install --upgrade pip && pip3 install -r /opt/requirements.txt

# Add main script
ADD main.py /usr/src/main.py

# Unzip model file
ADD efficientnetb0_v1_first_trial_4* /opt/
RUN cat /opt/efficientnetb0_v1_first_trial_4* >> /opt/efficientnetb0_v1_first_trial_4_combine.zip
CMD unzip /opt/efficientnetb0_v1_first_trial_4_combine.zip -d /usr/src/
RUN rm /opt/efficientnetb0_v1_first_trial_4*

# Add required dir
RUN mkdir -p /enigma/datasets/HA-Sample/
RUN mkdir -p /enigma/local_storage/result/
COPY . .
VOLUME ["/enigma", "/usr/src"]
