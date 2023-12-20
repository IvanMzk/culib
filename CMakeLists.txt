cmake_minimum_required(VERSION 3.5)
project(cuda_storage LANGUAGES CXX CUDA)

add_subdirectory(${${project_name}_SOURCE_DIR}//..//multithreading multithreading)
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    add_subdirectory(test)
endif()
add_subdirectory(src)