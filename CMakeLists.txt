cmake_minimum_required(VERSION 3.10)
project(slam_app)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE Debug)
add_definitions(-DEIGEN_MAX_ALIGN_BYTES=16)

# Find global dependencies
find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(pybind11 2.13 REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(g2o REQUIRED)
find_package(Pangolin REQUIRED)
find_package(DBoW3 REQUIRED)
find_package(Boost REQUIRED)


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
    pango_display
    pango_opengl
    DBoW3
)

target_include_directories(slam PRIVATE ${Boost_INCLUDE_DIRS})
# add_compile_definitions(EIGEN_MAX_STATIC_ALIGN_BYTES=0)
add_definitions(-msse -msse2 -msse3 -msse4.1 -msse4.2)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/atlas)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/feature_extractor)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/optimizer)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/depth_estimator/Stereo_Triangulation)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/visualize)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/interfaces)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/sensors)


