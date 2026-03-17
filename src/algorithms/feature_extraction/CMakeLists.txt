target_include_directories(slam PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

pybind11_add_module(FeatureExtractor SHARED ${CMAKE_CURRENT_SOURCE_DIR}/featureExtractor.cpp)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ORBExtractor)
target_link_libraries(FeatureExtractor PUBLIC 
    pybind11::module
    slam
)



