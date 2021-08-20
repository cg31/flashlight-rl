#!/bin/bash

# get/update arrayfire and flashlight

set -eux

TOP=$PWD

if [ ! -d $TOP/arrayfire ]; then
  git clone https://github.com/arrayfire/arrayfire.git
fi

cd $TOP/arrayfire

git pull
git submodule update --recursive --init
git prune

cd $TOP

if [ ! -d $TOP/flashlight ]; then
  git clone https://github.com/facebookresearch/flashlight.git
fi

cd $TOP/flashlight

git pull
git submodule update --recursive --init
git prune
