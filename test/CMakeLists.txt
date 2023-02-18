add_executable(Test)
target_link_libraries(Test PRIVATE cuda_experimental)
target_compile_features(Test PRIVATE cxx_std_17)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    target_compile_options(Test PRIVATE /W4 /arch:AVX2 /Zc:__cplusplus "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(Test PRIVATE -Wall "$<$<CONFIG:RELEASE>:-O3>")
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        target_compile_options(Test PRIVATE -stdlib=libc++)
    else()
    endif()
endif()


target_sources(Test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_pointer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_allocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_copy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_storage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_multi_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp
)