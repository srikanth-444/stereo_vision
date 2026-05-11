target_sources(slam PRIVATE
${CMAKE_CURRENT_SOURCE_DIR}/src/camera.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/imu.cpp
)

target_include_directories(slam PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

pybind11_add_module(Sensors SHARED
${CMAKE_CURRENT_SOURCE_DIR}/bindings.cpp
)
target_link_libraries(Sensors PUBLIC 
    pybind11::module
    slam
)
set_target_properties(Sensors PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

