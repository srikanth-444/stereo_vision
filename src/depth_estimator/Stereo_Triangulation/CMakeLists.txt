target_sources(slam PRIVATE
${CMAKE_CURRENT_SOURCE_DIR}/triangulation.cpp
)

target_include_directories(slam PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/
)

pybind11_add_module(Stereo SHARED
${CMAKE_CURRENT_SOURCE_DIR}/bindings.cpp
)
target_link_libraries(Stereo PUBLIC 
    pybind11::module
    slam
)
set_target_properties(Stereo PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)