FROM pytorch/pytorch:latest

WORKDIR /usr/src/
RUN apt update && gpt upgrade
RUN apt install python3 python3-pip
RUN pip install -r REQUIREMENT.txt
RUN mkdir -p /enigma/datasets/HA-Sample/
RUN mkdir -p /enigma/local_storage/result/
COPY . .
