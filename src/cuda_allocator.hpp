#ifndef CUDA_ALLOCATOR_HPP_
#define CUDA_ALLOCATOR_HPP_

#include "cuda_helpers.hpp"
#include "cuda_pointer.hpp"

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
            cuda_error_check(cudaFree(ptr_to_void(p)));
        }
    }
    bool operator==(const device_allocator& other)const{return true;}
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
            cuda_error_check(cudaFreeHost(ptr_to_void(p)));
        }
    }
    bool operator==(const locked_allocator& other)const{return flags == other.flags;}
};

/*
* allocate page-locked memory on host
*/
template<typename T>
class registered_allocator
{
    static_assert(std::is_trivially_copyable_v<T>);
    unsigned int flags;
public:
    using value_type = T;
    using pointer = locked_pointer<T>;
    using const_pointer = locked_pointer<const T>;
    using difference_type = typename pointer::difference_type;
    using size_type = difference_type;
    using is_aways_equal = std::true_type;

    explicit registered_allocator(unsigned int flags_ = cudaHostRegisterDefault):
        flags{flags_}
    {}
    pointer allocate(size_type n){
        unsigned char* p{nullptr};
        if (n){
            auto n_bytes = n*sizeof(value_type);
            p = new unsigned char[n_bytes];
            cuda_error_check(cudaHostRegister(p,n_bytes,flags));
        }
        return pointer{reinterpret_cast<T*>(p)};
    }
    void deallocate(pointer p, size_type){
        if (p){
            cuda_error_check(cudaHostUnregister(p.get()));
            delete[] reinterpret_cast<unsigned char*>(p.get());
        }
    }
    bool operator==(const registered_allocator& other)const{return flags == other.flags;}
};

/*
* allocate pageable memory on host
*/
template<typename T>
class pageable_allocator
{
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using difference_type = std::ptrdiff_t;
    using size_type = difference_type;

    pointer allocate(size_type n){
        void* p{nullptr};
        if (n){
            p = new char[n*sizeof(T)];
        }
        return static_cast<pointer>(p);
    }
    void deallocate(pointer p, size_type){
        if (p){
            delete[] reinterpret_cast<char*>(p);
        }
    }
    bool operator==(const pageable_allocator& other)const{return true;}
};


}   //end of namespace cuda_experimental

#endif