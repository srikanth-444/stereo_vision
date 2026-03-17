
target_sources(slam PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/orb.cpp
)

target_include_directories(slam PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_sources(FeatureExtractor PRIVATE
${CMAKE_CURRENT_SOURCE_DIR}/bindings/orb_py.cpp
)


