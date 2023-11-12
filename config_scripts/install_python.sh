#!/bin/bash

wget "https://www.python.org/ftp/python/3.8.17/Python-3.8.17.tgz"
tar -zxvf Python-3.8.17.tgz Python-3.8.17/
cd Python-3.8.17/
./configure --prefix=/usr
make
sudo make install
