cmake_minimum_required(VERSION 3.9)
project(culib)
enable_language(CUDA)

set(include_dir ${CMAKE_CURRENT_LIST_DIR}/include)
set(src_dir ${CMAKE_CURRENT_LIST_DIR}/src)
set(cmake_dir ${CMAKE_CURRENT_LIST_DIR}/cmake)

find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
include(${cmake_dir}/FindAvxDetect.cmake)
add_subdirectory(${src_dir})

#helper target to propagate common settings
add_library(common_config_ INTERFACE)
target_compile_features(common_config_ INTERFACE cxx_std_17)
target_include_directories(common_config_
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>
    INTERFACE $<BUILD_INTERFACE:${include_dir}>
)
target_link_libraries(common_config_
    INTERFACE ${CUDART_LIBRARY}
    INTERFACE AvxDetect::AvxDetect
)

add_library(culib INTERFACE)
target_link_libraries(culib
    INTERFACE common_config_
    INTERFACE culib_bin
)

option(BUILD_TEST "build culib tests" OFF)
option(BUILD_BENCHMARK "build culib benchmark" OFF)

if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    if(BUILD_TEST)
        add_subdirectory(test)
    endif()
    if(BUILD_BENCHMARK)
        add_subdirectory(benchmark)
    endif()
endif()
