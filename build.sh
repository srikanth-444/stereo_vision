#!/bin/bash

set -e

echo "Starting build process..."


BUILD_DIR="build"
SOURCE_DIR="."

echo "cleaning build files"

rm -rf $BUILD_DIR
mkdir $BUILD_DIR

echo "--- Running CMake Configuration ---"
cmake -S $SOURCE_DIR -B $BUILD_DIR

echo "--- Starting Compilation ---"
cmake --build $BUILD_DIR -j $(nproc)

echo "--- Build Successful! ---"

echo "--- Initiating Smoke Tests"
pytest -v