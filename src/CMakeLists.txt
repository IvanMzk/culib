add_library(cuda_experimental OBJECT)
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
target_compile_features(cuda_experimental PUBLIC cxx_std_17)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    target_compile_options(cuda_experimental PUBLIC /W4 /arch:AVX2 /Zc:__cplusplus "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(cuda_experimental PUBLIC -Wall -mavx2 "$<$<CONFIG:RELEASE>:-O3>")
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        target_compile_options(cuda_experimental PUBLIC -stdlib=libc++)
    else()
    endif()
endif()
