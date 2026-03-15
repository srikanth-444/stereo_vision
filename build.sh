#!/bin/bash

set -e

echo "Starting build process..."

Build_DIR="src/algorithms/feature_extraction/Orb_slam_extractor/build"
SOURCE_DIR="src/algorithms/feature_extraction/Orb_slam_extractor/src"

echo "cleaning build files"
rm -rf $BUILD_DIR
mkdir $BUILD_DIR

echo "--- Running CMake Configuration ---"
cmake -S $SOURCE_DIR -B $BUILD_DIR

echo "--- Starting Compilation ---"
cmake --build $BUILD_DIR -j $(nproc)

echo "--- Build Successful! ---"