add_library(culib_bin STATIC)
target_link_libraries(culib_bin common_config_)
target_sources(culib_bin PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/cuda_helpers.cpp
)

if (AvxDetect_HAS_AVX2)
    target_sources(culib_bin PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/memcpy_avx.cpp
    )
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    target_compile_options(culib_bin PRIVATE /W4 /arch:AVX2 /Zc:__cplusplus "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(culib_bin PRIVATE -Wall -mavx2 "$<$<CONFIG:RELEASE>:-O2>")
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    else()
    endif()
endif()


# add_library(culib OBJECT)
# target_include_directories(culib PUBLIC
#     ${${project_name}_SOURCE_DIR}\\src
#     C://Program\ Files//NVIDIA\ GPU\ Computing\ Toolkit//CUDA//v10.2//include
# )
# target_link_directories(culib PUBLIC C://Program\ Files//NVIDIA\ GPU\ Computing\ Toolkit//CUDA//v10.2//lib//x64)
# target_sources(culib PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/cuda_helpers.cpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/cuda_copy.cpp
# )
# target_link_libraries(culib PUBLIC
#     multithreading
#     cudart_static.lib
# )
# target_compile_features(culib PUBLIC cxx_std_17)
# if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
#     target_compile_options(culib PUBLIC /W4 /arch:AVX2 /Zc:__cplusplus "$<$<CONFIG:RELEASE>:/O2>")
# else()
#     target_compile_options(culib PUBLIC -Wall -mavx2 "$<$<CONFIG:RELEASE>:-O3>")
#     if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
#     else()
#     endif()
# endif()
