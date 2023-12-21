add_executable(Benchmark)
target_include_directories(Benchmark PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../qa_common)
target_link_libraries(Benchmark PRIVATE culib)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    target_compile_options(Benchmark PRIVATE /W4 /Zc:__cplusplus "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(Benchmark PRIVATE -Wall "$<$<CONFIG:RELEASE>:-O2>")
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    else()
    endif()
endif()

target_sources(Benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_cuda_copy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/msvc_fake_use.cpp
)