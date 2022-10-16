#ifndef DEVICE_STORAGE_HPP_
#define DEVICE_STORAGE_HPP_

#include "cuda_runtime.h"
#include "cuda_helpers.hpp"

namespace cuda_experimental{

template<typename T>
class cuda_allocator
{
public:
    using size_type = std::size_t;
    using value_type = T;
    using pointer = T*;

    pointer allocate(size_type n){
        void* p;
        cuda_error_check(cudaMalloc(&p,n*sizeof(T)));
        return reinterpret_cast<pointer>(p);
    }

    void deallocate(pointer p, size_type n){
        cuda_error_check(cudaFree(static_cast<void*>(p)));
    }
};



template<typename T, typename Alloc = cuda_allocator<T>>
class device_storage : private Alloc
{
    using allocator_type = Alloc;
    using value_type = typename allocator_type::value_type;
    using pointer = typename allocator_type::pointer;
    using size_type = typename allocator_type::size_type;

    pointer begin_;
    size_type size_;

    pointer allocate(const size_type& n){
        return allocator_type::allocate(n);
    }
    void deallocate(){
        if (begin_){
            allocator_type::deallocate(begin_,size_);
            begin_ = nullptr;
        }
    }

    device_storage(const device_storage& other) = delete;

public:
    ~device_storage(){deallocate();}
    device_storage(device_storage&& other):
        begin_{other.begin_},
        size_{other.size_}
    {
        other.begin_ = nullptr;
    }
    device_storage(const size_type& n):
        begin_{allocate(n)},
        size_{n}
    {}

    auto size()const{return size_;}
    auto begin(){return begin_;}

};

}   //end of namespace cuda_experimental



#endif