cmake_minimum_required(VERSION 3.10)
project(orb_slam_features)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # Needed for Python extensions

find_package(OpenCV REQUIRED)
find_package(pybind11 REQUIRED)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")
# Include your ORB-SLAM3 source files
add_library(orb_slam_features SHARED 
${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp 
${CMAKE_CURRENT_SOURCE_DIR}/src/orb.cpp
)


# Correct target_include_directories syntax
target_include_directories(orb_slam_features PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(orb_slam_features PRIVATE
    ${OpenCV_LIBS}
    pybind11::module
)

# Ensure correct Python module naming
set_target_properties(orb_slam_features PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)