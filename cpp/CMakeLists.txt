cmake_minimum_required(VERSION 3.28)
project(fastmap)

set(CMAKE_CXX_STANDARD 20)

add_library(fastmap SHARED library.cpp
        mymath.hpp
        segment.h
        segment.cpp
        graph_decoder.h
        graph_decoder.cpp)

install(TARGETS fastmap
    LIBRARY DESTINATION fastmap/release
    RUNTIME DESTINATION fastmap/release
)

if(NOT SKBUILD)
    set_target_properties(fastmap PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../src/fastmap/release"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../src/fastmap/release"
    )
endif()
