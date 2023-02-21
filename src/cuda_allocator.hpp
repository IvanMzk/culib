#ifndef CUDA_ALLOCATOR_HPP_
#define CUDA_ALLOCATOR_HPP_

#include "cuda_helpers.hpp"

namespace cuda_experimental{

/*
* allocate memory on current active device
*/
template<typename T>
class device_allocator
{
    static_assert(std::is_trivially_copyable_v<T>);
public:
    using value_type = T;
    using pointer = device_pointer<T>;
    using const_pointer = device_pointer<const T>;
    using difference_type = typename pointer::difference_type;
    using size_type = difference_type;
    //using size_type = std::size_t;

    pointer allocate(size_type n){
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
};

/*
* allocate page-locked memory on host
*/
template<typename T>
class locked_allocator
{
    static_assert(std::is_trivially_copyable_v<T>);
    unsigned int flags;
public:
    using value_type = T;
    using pointer = locked_pointer<T>;
    using const_pointer = locked_pointer<const T>;
    using difference_type = typename pointer::difference_type;
    using size_type = difference_type;
    //using size_type = std::size_t;
    using is_aways_equal = std::true_type;

    explicit locked_allocator(unsigned int flags_ = cudaHostAllocDefault):
        flags{flags_}
    {}
    pointer allocate(size_type n){
        void* p{nullptr};
        if (n){
            cuda_error_check(cudaHostAlloc(&p,n*sizeof(T),flags));
        }
        return pointer{static_cast<T*>(p)};
    }
    void deallocate(pointer p, size_type){
        if (p){
            cuda_error_check(cudaFreeHost(p));
        }
    }
    bool operator==(const locked_allocator& other)const{return flags == other.flags;}
};

}   //end of namespace cuda_experimental

#endif