add_executable(Test)
target_include_directories(Test PRIVATE ${${project_name}_SOURCE_DIR}//src)
target_compile_options(Test PRIVATE -W1 -std:c++17 /Zc:__cplusplus)

add_executable(Test_selected)
target_include_directories(Test_selected PRIVATE ${${project_name}_SOURCE_DIR}//src)
target_include_directories(Test_selected PRIVATE C://Program\ Files//NVIDIA\ GPU\ Computing\ Toolkit//CUDA//v10.2//include)
target_link_directories(Test_selected PRIVATE C://Program\ Files//NVIDIA\ GPU\ Computing\ Toolkit//CUDA//v10.2//lib//x64)
target_compile_options(Test_selected PRIVATE -W1 -std:c++17 /Zc:__cplusplus "$<$<CONFIG:RELEASE>:-O2>" /bigobj /MP)
target_link_libraries(Test_selected PRIVATE cudart_static.lib)
target_link_options(Test_selected PRIVATE /STACK:104857600)

target_sources(Test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/test_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp
)

target_sources(Test_selected PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/test_device_storage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_helpers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp
)