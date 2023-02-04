add_executable(Test)
target_include_directories(Test PRIVATE ${${project_name}_SOURCE_DIR}//src)
target_compile_options(Test PRIVATE -W1 -std:c++17 /Zc:__cplusplus)
target_sources(Test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_helpers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_cuda_copy_experiment.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_cuda_copy.cpp
)



add_executable(Test_selected)
target_link_libraries(Test_selected
    PRIVATE cuda_experimental
)
target_link_options(Test_selected PRIVATE /STACK:104857600)


target_sources(Test_selected PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_pointer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_allocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_copy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_storage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_multi_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp
)