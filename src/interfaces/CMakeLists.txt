target_sources(slam PRIVATE
${CMAKE_CURRENT_SOURCE_DIR}/src/camera_csv_source.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/imu_csv_source.cpp
)

target_include_directories(slam PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

pybind11_add_module(Interface SHARED
${CMAKE_CURRENT_SOURCE_DIR}/bindings.cpp
)
target_link_libraries(Interface PUBLIC 
    pybind11::module
    slam
)
set_target_properties(Interface PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)