FROM ubuntu:14.04

# Install basic applications
RUN apt-get update && apt-get install -y awscli python python-dev python-distribute python-pip python-numpy python-scipy python-gevent python-matplotlib 

RUN sudo apt-get install libblas3gf libopenblas-base libatlas3gf-base liblapack3gf libopenblas-dev 

# Install python dependencies for next_backend
ADD requirements.txt /requirements.txt
RUN pip install -r requirements.txt

