# GTensor - computation library
# Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
#
# Distributed under the Boost Software License, Version 1.0.
# The full license is in the file LICENSE.txt, distributed with this software.

#FindAvxDetect module determines the presence of AVX, FMA on the system
#This module defines AvxDetect::AvxDetect imported target

include(CheckCXXSourceRuns)

set (AVX_DETECT_SOURCE [[
#include <immintrin.h>
int main(){
    const float src = 2.0f;
    const float expected = 4.0f;
    float dst[8];
    auto y = _mm256_broadcast_ss(&src);
    _mm256_storeu_ps(dst,_mm256_add_ps(y,y));
    for (auto i=0; i!=8; ++i){
        if (dst[i]!=expected){
            return 1;
        }
    }
    return 0;
}
]])

set (AVX2_DETECT_SOURCE [[
#include <immintrin.h>
int main(){
    const float src[8]={1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0};
    float dst[8];
    auto y = _mm256_loadu_ps(src);
    _mm256_storeu_ps(dst,_mm256_permutevar8x32_ps(y,_mm256_set_epi32(0,1,2,3,4,5,6,7)));
    for (auto i=0; i!=8; ++i){
        if (dst[i]!=src[7-i]){
            return 1;
        }
    }
    return 0;
}
]])

set (FMA_DETECT_SOURCE [[
#include <immintrin.h>
int main(){
    const float src = 2.0f;
    const float expected = 6.0f;
    float dst[8];
    auto y = _mm256_broadcast_ss(&src);
    _mm256_storeu_ps(dst,_mm256_fmadd_ps(y,y,y));
    for (auto i=0; i!=8; ++i){
        if (dst[i]!=expected){
            return 1;
        }
    }
    return 0;
}
]])

add_library(AvxDetect::AvxDetect INTERFACE IMPORTED)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(CMAKE_REQUIRED_FLAGS "/arch:AVX")
    check_cxx_source_runs("${AVX_DETECT_SOURCE}" AvxDetect_HAS_AVX)
    set(CMAKE_REQUIRED_FLAGS "/arch:AVX2")
    check_cxx_source_runs("${AVX2_DETECT_SOURCE}" AvxDetect_HAS_AVX2)
    set(CMAKE_REQUIRED_FLAGS "/arch:AVX2")
    check_cxx_source_runs("${FMA_DETECT_SOURCE}" AvxDetect_HAS_FMA)
    if (AvxDetect_HAS_AVX2 OR AvxDetect_HAS_FMA)
        target_compile_options(AvxDetect::AvxDetect INTERFACE /arch:AVX2)
    elseif(AvxDetect_HAS_AVX)
        target_compile_options(AvxDetect::AvxDetect INTERFACE /arch:AVX)
    endif()
else()
    set(CMAKE_REQUIRED_FLAGS "-mavx")
    check_cxx_source_runs("${AVX_DETECT_SOURCE}" AvxDetect_HAS_AVX)
    set(CMAKE_REQUIRED_FLAGS "-mavx2")
    check_cxx_source_runs("${AVX2_DETECT_SOURCE}" AvxDetect_HAS_AVX2)
    set(CMAKE_REQUIRED_FLAGS "-mfma")
    check_cxx_source_runs("${FMA_DETECT_SOURCE}" AvxDetect_HAS_FMA)
    if (AvxDetect_HAS_AVX)
        target_compile_options(AvxDetect::AvxDetect INTERFACE -mavx)
    endif()
    if (AvxDetect_HAS_AVX2)
        target_compile_options(AvxDetect::AvxDetect INTERFACE -mavx2)
    endif()
    if (AvxDetect_HAS_FMA)
        target_compile_options(AvxDetect::AvxDetect INTERFACE -mfma)
    endif()
endif()