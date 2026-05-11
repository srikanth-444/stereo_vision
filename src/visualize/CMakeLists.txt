target_sources(slam PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/visualize.cpp)

target_include_directories(slam PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

pybind11_add_module(Visualize SHARED
${CMAKE_CURRENT_SOURCE_DIR}/visualize_py.cpp
)
target_link_libraries(Visualize PUBLIC 
    pybind11::module
    slam
)
set_target_properties(Visualize PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
