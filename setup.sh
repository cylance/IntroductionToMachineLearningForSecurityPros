#!/usr/bin/env bash
sudo apt-get update;
sudo apt-get upgrade -y;

sudo apt-get install python-dev python-pip git libhdf5-dev python-tk -y

sudo pip install numpy scipy sklearn keras theano tensorflow h5py matplotlib

git clone https://github.com/CylanceSPEAR/IntroductionToMachineLearningForSecurityPros.git
