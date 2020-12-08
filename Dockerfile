FROM pytorch/pytorch:latest

WORKDIR /usr/src/
USER root

# Set Environment Variable
ENV LC_ALL=C.UTF-8

RUN apt-get -y update
RUN set -x; apt-get install -y --no-install-recommends p7zip-full python3 python3-pip libgl1-mesa-glx libgtk2.0-dev
 
# Install dependencies
ADD REQUIREMENT.txt /opt/requirements.txt
RUN python3 -m pip install --upgrade pip && pip3 install -r /opt/requirements.txt

# Add main script
ADD main.py /usr/src/main.py
RUN mkdir /root/submission
ADD main.py /root/submission/main.py

# Unzip model file
RUN mkdir /opt/model
ADD resnet50_first_trial_maxacc* /opt/model/
RUN cd /usr/src && 7z x /opt/model/resnet50_first_trial_maxacc.zip
RUN rm -rf /opt/model

# Add required dir
RUN mkdir -p /enigma/datasets/HA-Sample/
RUN mkdir -p /enigma/local_storage/result/
COPY . .
VOLUME ["/enigma", "/usr/src"]
