#ifndef CUDA_MEMORY_HPP_
#define CUDA_MEMORY_HPP_

#include "cuda_helpers.hpp"

namespace cuda_experimental{

template<typename T>
class cuda_pointer{
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;

    const_pointer ptr;
public:
    cuda_pointer() = default;
    cuda_pointer(const cuda_pointer&) = default;
    cuda_pointer& operator=(const cuda_pointer&) = default;
    explicit cuda_pointer(const_pointer ptr_):
        ptr{ptr_}
    {}
    cuda_pointer& operator=(std::nullptr_t){
        ptr = nullptr;
        return *this;
    }

    operator bool()const{return static_cast<bool>(ptr);}
    pointer get(){return const_cast<pointer>(ptr);}
    const_pointer get()const{return ptr;}
};

template<typename T>
auto operator==(const cuda_pointer<T>& lhs, const cuda_pointer<T>& rhs){return lhs.get() == rhs.get();}
template<typename T>
auto operator!=(const cuda_pointer<T>& lhs, const cuda_pointer<T>& rhs){return !(lhs == rhs);}
template<typename T>
auto operator-(const cuda_pointer<T>& lhs, const cuda_pointer<T>& rhs){return lhs.get() - rhs.get();}
template<typename T, typename U>
auto operator+(const cuda_pointer<T>& lhs, const U& rhs){return cuda_pointer<T>{lhs.get() + rhs};}
template<typename T, typename U>
auto operator+(const U& lhs, const cuda_pointer<T>& rhs){return rhs+lhs;}
template<typename T, typename U>
auto operator-(const cuda_pointer<T>& lhs, const U& rhs){return lhs+-rhs;}
template<typename T>
auto distance(const cuda_pointer<T>& begin, const cuda_pointer<T>& end){return end-begin;}
template<typename T>
auto ptr_to_void(const cuda_pointer<T>& p){return static_cast<const void*>(p.get());}
template<typename T>
auto ptr_to_void(cuda_pointer<T>& p){return static_cast<void*>(p.get());}
template<typename T>
auto ptr_to_void(const T* p){return static_cast<const void*>(p);}
template<typename T>
auto ptr_to_void(T* p){return static_cast<void*>(p);}


template<typename T>
class cuda_allocator
{
public:
    using difference_type = std::ptrdiff_t;
    using size_type = difference_type;
    using value_type = T;
    using pointer = cuda_pointer<T>;
    using const_pointer = const cuda_pointer<T>;

    pointer allocate(size_type n){
        void* p;
        cuda_error_check(cudaMalloc(&p,n*sizeof(T)));
        return pointer{reinterpret_cast<T*>(p)};
    }
    void deallocate(pointer p, size_type n){
        cuda_error_check(cudaFree(static_cast<void*>(p.get())));
    }
};

template<typename T>
class cuda_memory_handler : public cuda_allocator<T>
{
public:
    using typename cuda_allocator::size_type;
    using typename cuda_allocator::value_type;
    using cuda_pointer = typename cuda_allocator::pointer;
    using const_cuda_pointer = typename cuda_allocator::const_pointer;
    using host_pointer = T*;
    using const_host_pointer = const T*;

    void memcpy_host_to_device(cuda_pointer device_dst, const_host_pointer host_src, size_type n){
        cuda_error_check(cudaMemcpy(ptr_to_void(device_dst), ptr_to_void(host_src), n*sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice));
    }
    void memcpy_device_to_host(host_pointer host_dst, const_cuda_pointer device_src, size_type n){
        cuda_error_check(cudaMemcpy(ptr_to_void(host_src), ptr_to_void(device_dst), n*sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    }
    void memcpy_device_to_device(cuda_pointer device_dst, const_cuda_pointer device_src, size_type n){
        cuda_error_check(cudaMemcpy(ptr_to_void(device_dst), ptr_to_void(device_dst), n*sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToDevice));
    }
};

}   //end of namespace cuda_experimental

#endif