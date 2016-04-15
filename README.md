# cifar.torch

For original readme, please see https://github.com/szagoruyko/cifar.torch , and Sergey's blog post at http://torch.ch/blog/2015/07/30/cifar.html

This readme is for pytorch version, which handles data loading and preprocessing in python

## Installation

- If using CUDA, you need at least CUDA 7.0 and CuDNN v4
- Install Torch:
```
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch
# install dependencies.  To install everything:
  bash install-deps
# Or, if you're on ubuntu, you only need the following dependencies:
  sudo apt-get update -y
  sudo apt-get install -y wget git gcc g++ cmake libffi-dev \
       libblas-dev liblapack-dev libatlas-base-dev gfortran libreadline-dev
# install torch
./install.sh
```
- install torch cudnn (only if using CUDA) and nninit:
```
luarocks install cudnn
luarocks install nninit
```
- Setup python, eg for python 3.4:
```
sudo apt-get install python3.4-dev
virtualenv -p python3.4 ~/env34
source ~/env34/bin/activate
pip install docopt
pip install numpy
```
- Install pytorch:
```
git clone https://github.com/hughperkins/pytorch ~/pytorch
cd ~/pytorch
source ~/torch/install/bin/torch-activate
./build.sh
```

## Data download

```bash
./download-cifar.sh
```

## Training

```bash
python train.py
```

You should see the loss gradually decrease, and the test accuracy gradually decrease.

# Differences from original lua version

- data loading in python
- preprocessing in python
- no conversion from rgb to yuv (just because... haven't added it)
- no graph for now (but... it's python... you can use all the matplot goodness you are used to using :-) )

