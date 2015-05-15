#!/usr/bin/env bash

apt-get update

apt-get install -y python-pip

apt-get install -y linux-image-extra-$(uname -r) aufs-tools

curl -sSL https://get.docker.com/ubuntu/ | sh

easy_install -U pip

pip install requests==2.5.2

pip install docker-compose
