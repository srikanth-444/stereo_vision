cmake_minimum_required(VERSION 3.10)
project(slam_app)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find global dependencies
find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(pybind11 REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

message(STATUS "OpenCV includes: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Eigen includes: ${EIGEN3_INCLUDE_DIRS}")

add_library(slam SHARED)
target_link_libraries(slam PUBLIC
    ${OpenCV_LIBS}
    Eigen3::Eigen
)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/maper)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/feature_extraction)