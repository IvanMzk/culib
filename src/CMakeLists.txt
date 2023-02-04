add_library(cuda_experimental OBJECT)
target_compile_options(cuda_experimental PUBLIC -W1 -std:c++17 /arch:AVX2 /Zc:__cplusplus "$<$<CONFIG:RELEASE>:-O2>" /bigobj /MP)
target_include_directories(cuda_experimental PUBLIC
    ${${project_name}_SOURCE_DIR}\\src
    C://Program\ Files//NVIDIA\ GPU\ Computing\ Toolkit//CUDA//v10.2//include
)
target_link_directories(cuda_experimental PUBLIC C://Program\ Files//NVIDIA\ GPU\ Computing\ Toolkit//CUDA//v10.2//lib//x64)
target_sources(cuda_experimental PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_helpers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_copy.cpp
)
target_link_libraries(cuda_experimental PUBLIC
    multithreading
    cudart_static.lib
)
