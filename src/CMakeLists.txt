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
    target_compile_options(culib_bin PRIVATE /W4 /Zc:__cplusplus "$<$<CONFIG:RELEASE>:/O2>")
else()
    target_compile_options(culib_bin PRIVATE -Wall "$<$<CONFIG:RELEASE>:-O2>")
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    else()
    endif()
endif()
