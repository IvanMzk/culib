add_executable(Test)
target_include_directories(Test PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../qa_common)
target_link_libraries(Test PRIVATE culib)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    target_compile_options(Test PRIVATE /W4 /Zc:__cplusplus "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(Test PRIVATE -Wall "$<$<CONFIG:RELEASE>:-O2>")
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    else()
    endif()
endif()

target_sources(Test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_helpers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_pointer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_allocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_copy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_storage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_multi_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp
)

if (AvxDetect_HAS_AVX2)
    target_sources(Test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/test_memcpy_avx.cpp
    )
endif()