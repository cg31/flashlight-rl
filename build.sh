#!/bin/bash

set -eux

TOP=$PWD

export CFLAGS="-Wno-sign-compare -Wno-deprecated-declarations"
export CXXFLAGS=$CFLAGS

cd $TOP/arrayfire

#rm -rf build
mkdir -p build
cd build

cmake -GNinja -DCMAKE_MESSAGE_LOG_LEVEL=ERROR -Wno-dev -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=. \
-DAF_BUILD_CUDA=ON -DAF_BUILD_OPENCL=OFF -DAF_BUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF \
-DCMAKE_C_FLAGS="$CFLAGS" -DCMAKE_CXX_FLAGS="$CXXFLAGS" \
..

ninja -j 32
ninja -j 32 install

cd $TOP/flashlight

#rm -rf build
mkdir -p build
cd build

cmake -GNinja -DCMAKE_MESSAGE_LOG_LEVEL=ERROR  -Wno-dev -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=. \
-DFL_BACKEND="CUDA" -DCMAKE_PREFIX_PATH="$TOP/arrayfire/build" \
-DFL_BUILD_TESTS=OFF -DFL_BUILD_EXAMPLES=OFF \
-DCMAKE_C_FLAGS="$CFLAGS" -DCMAKE_CXX_FLAGS="$CXXFLAGS" \
..

ninja -j 32
ninja -j 32 install


cd $TOP

#rm -rf build
mkdir -p build/dbg
cd build/dbg

cmake -GNinja -DCMAKE_MESSAGE_LOG_LEVEL=ERROR -DCMAKE_BUILD_TYPE=Debug \
-DCMAKE_C_FLAGS="$CFLAGS" -DCMAKE_CXX_FLAGS="$CXXFLAGS" \
../..

cmake --build . --parallel

cd $TOP

#rm -rf build
mkdir -p build/dbg
cd build/dbg

cmake -GNinja -DCMAKE_MESSAGE_LOG_LEVEL=ERROR -DCMAKE_BUILD_TYPE=Release \
-DCMAKE_C_FLAGS="$CFLAGS" -DCMAKE_CXX_FLAGS="$CXXFLAGS" \
../..

cmake --build . --parallel
