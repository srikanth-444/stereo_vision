target_sources(slam PRIVATE
${CMAKE_CURRENT_SOURCE_DIR}/src/frame.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/landmark.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/map.cpp
)

target_include_directories(slam PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

pybind11_add_module(Atlas SHARED
${CMAKE_CURRENT_SOURCE_DIR}/bindings/atlas.cpp
${CMAKE_CURRENT_SOURCE_DIR}/bindings/frame_py.cpp
${CMAKE_CURRENT_SOURCE_DIR}/bindings/landmark_py.cpp
${CMAKE_CURRENT_SOURCE_DIR}/bindings/map_py.cpp
)
target_link_libraries(Atlas PUBLIC 
    pybind11::module
    slam
)
set_target_properties(Atlas PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
