/*
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#ifndef CUDA_ALLOCATOR_HPP_
#define CUDA_ALLOCATOR_HPP_

#include "cuda_helpers.hpp"

namespace culib{

//allocate memory on current active device
template<typename T>
class device_allocator
{
    static_assert(std::is_trivially_copyable_v<T>);
public:
    using value_type = T;
    using pointer = device_pointer<T>;
    using const_pointer = device_pointer<const T>;
    using difference_type = typename std::iterator_traits<pointer>::difference_type;
    using size_type = std::size_t;
    using is_aways_equal = std::true_type;

    template<typename U> struct rebind{using other = device_allocator<U>;};

    device_allocator() noexcept
    {}
    device_allocator(const device_allocator&) noexcept
    {}
    template<typename U>
    device_allocator(const device_allocator<U>&) noexcept
    {}

    pointer allocate(const size_type n){
        void* p{nullptr};
        if (n){
            cuda_error_check(cudaMalloc(&p,n*sizeof(T)));
        }
        return pointer{static_cast<T*>(p), cuda_get_device()};
    }
    void deallocate(pointer p, size_type){
        if (p){
            device_switcher switcher{p.device()};
            cuda_error_check(cudaFree(p));
        }
    }
    bool operator==(const device_allocator&)const{return true;}
    bool operator!=(const device_allocator&)const{return false;}
};

//allocate page-locked memory on host
template<typename T>
class locked_allocator
{
    static_assert(std::is_trivially_copyable_v<T>);
public:
    using value_type = T;
    using pointer = locked_pointer<T>;
    using const_pointer = locked_pointer<const T>;
    using difference_type = typename std::iterator_traits<pointer>::difference_type;
    using size_type = std::size_t;
    using is_aways_equal = std::true_type;

    template<typename U> struct rebind{using other = locked_allocator<U>;};

    locked_allocator() noexcept
    {}
    locked_allocator(const locked_allocator&) noexcept
    {}
    template<typename U>
    locked_allocator(const locked_allocator<U>&) noexcept
    {}

    pointer allocate(size_type n){
        void* p{nullptr};
        if (n){
            cuda_error_check(cudaHostAlloc(&p,n*sizeof(T),cudaHostAllocDefault));
        }
        return pointer{static_cast<T*>(p)};
    }
    void deallocate(pointer p, size_type){
        if (p){
            cuda_error_check(cudaFreeHost(p));
        }
    }
    bool operator==(const locked_allocator& other)const{return true;}
    bool operator!=(const locked_allocator& other)const{return false;}
};

}   //end of namespace culib
#endif