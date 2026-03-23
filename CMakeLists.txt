cmake_minimum_required(VERSION 3.10)
project(slam_app)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find global dependencies
find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(pybind11 2.13 REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(g2o REQUIRED)


message(STATUS "OpenCV includes: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Eigen includes: ${EIGEN3_INCLUDE_DIRS}")

add_library(slam SHARED)
link_directories(../thirdparty/g2o/build/lib)
target_link_libraries(slam PUBLIC
    ${OpenCV_LIBS}
    Eigen3::Eigen
    g2o_core
    g2o_stuff
    g2o_types_slam3d
    g2o_solver_csparse
    g2o_types_sba 
)
add_compile_definitions(EIGEN_MAX_STATIC_ALIGN_BYTES=0)
add_compile_definitions(EIGEN_DONT_VECTORIZE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/atlas)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/feature_extractor)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/optimizer)

