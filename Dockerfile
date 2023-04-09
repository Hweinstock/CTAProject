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

# RUN \
# pip install yfinance && \
# pip install benzinga && \
# pip install python-dotenv && \ 
# pip install colorama && \
# pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu && \
# pip install colorama && \
# pip install transformers

RUN \ 
pip install python-dotenv==1.0.0 && \
pip install torch==1.13.1 && \
pip install yfinance==0.2.12 && \
pip install benzinga==1.21 && \
pip install transformers==4.27.3 && \
pip install colorama==0.4.6 && \
pip install scikit-learn==1.2.2 && \
pip install Backtesting==0.3.3

WORKDIR /project/
