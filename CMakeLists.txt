cmake_minimum_required(VERSION 3.10)

project(stereo_Vision)

set(OpenCV_DIR D:/Installed softwares/opencv/build)

find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} src/main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})