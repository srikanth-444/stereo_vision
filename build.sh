#!/bin/bash

set -e

echo "Starting build process..."


BUILD_DIR="build"
SOURCE_DIR="."

echo "cleaning build files"

rm -rf $BUILD_DIR
mkdir $BUILD_DIR

echo "--- Running CMake Configuration ---"
cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" -DCMAKE_PREFIX_PATH="$HOME/.local/lib/python3.10/site-packages;/usr/local/lib"


echo "--- Starting Compilation ---"
cmake --build $BUILD_DIR -j $(nproc)

echo "--- Build Successful! ---"

echo "--- Initiating Smoke Tests"
export LD_LIBRARY_PATH=/usr/lib:/usr/local/lib:$LD_LIBRARY_PATH
pytest -v