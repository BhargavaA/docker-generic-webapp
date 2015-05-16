FROM ubuntu:14.04

# Install basic applications
RUN apt-get update && apt-get install -y \
    python \
    python-dev \
    python-distribute \ 
    python-pip \
    python-numpy \
    python-scipy \
    python-gevent 

# Install python dependencies for next_backend
ADD requirements.txt /requirements.txt
RUN pip install -U -r requirements.txt

# Expose port 8000 to the host
EXPOSE 8000



