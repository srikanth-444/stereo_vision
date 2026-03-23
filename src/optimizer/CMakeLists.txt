target_sources(slam PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/optimizer.cpp)

target_include_directories(slam PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

pybind11_add_module(Optimizer SHARED
${CMAKE_CURRENT_SOURCE_DIR}/binding.cpp
)
target_link_libraries(Optimizer PUBLIC 
    pybind11::module
    slam
)
set_target_properties(Optimizer PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
