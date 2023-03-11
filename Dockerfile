FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN \
apt-get update && \ 
apt-get upgrade -y && \
apt-get install -y vim && \
apt-get install -y sudo && \
apt-get install -y cmake && \
apt-get install -y gcc && \ 
apt-get install -y software-properties-common && \ 
apt-get install -y python3 && \ 
apt-get install -y python3-pip

RUN \
pip install yfinance && \
pip install benzinga && \
pip install python-dotenv && \ 
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

WORKDIR /project/

