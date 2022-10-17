#ifndef CUDA_MEMORY_HPP_
#define CUDA_MEMORY_HPP_

#include "cuda_helpers.hpp"

namespace cuda_experimental{

template<typename T, typename DiffT>
class cuda_pointer{
    using difference_type = DiffT;
    using value_type = T;
    using pointer = T*;

    pointer ptr;
public:
    cuda_pointer() = default;
    explicit cuda_pointer(pointer ptr_):
        ptr{ptr_}
    {}
    cuda_pointer& operator=(const cuda_pointer& other){
        ptr = other.ptr;
        return *this;
    }
    cuda_pointer& operator=(std::nullptr_t){
        ptr = nullptr;
        return *this;
    }

    operator bool()const{return static_cast<bool>(ptr);}
    auto get()const{return ptr;}
};

template<typename T, typename DiffT>
auto operator==(const cuda_pointer<T,DiffT>& lhs, const cuda_pointer<T,DiffT>& rhs){return lhs.get() == rhs.get();}
template<typename T, typename DiffT>
auto operator!=(const cuda_pointer<T,DiffT>& lhs, const cuda_pointer<T,DiffT>& rhs){return !(lhs == rhs);}
template<typename T, typename DiffT>
auto operator-(const cuda_pointer<T,DiffT>& lhs, const cuda_pointer<T,DiffT>& rhs){return lhs.get() - rhs.get();}
template<typename T, typename DiffT>
auto operator+(const cuda_pointer<T,DiffT>& lhs, const DiffT& rhs){return cuda_pointer<T,DiffT>{lhs.get() + rhs};}
template<typename T, typename DiffT>
auto operator+(const DiffT& lhs, const cuda_pointer<T,DiffT>& rhs){return rhs+lhs;}
template<typename T, typename DiffT>
auto distance(const cuda_pointer<T,DiffT>& begin, const cuda_pointer<T,DiffT>& end){return end-begin;}

template<typename T>
class cuda_allocator
{
public:
    using difference_type = std::ptrdiff_t;
    using size_type = difference_type;
    using value_type = T;
    using pointer = cuda_pointer<T,difference_type>;

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
    using host_pointer
public:
    using typename cuda_allocator::size_type;
    using typename cuda_allocator::value_type;
    using typename cuda_allocator::pointer;

    void memcpy_host_to_device(pointer device_dst, pointer host_src, size_type n){
        cuda_error_check(cudaMemcpy(static_cast<void*>(device_dst.get()), static_cast<void*>(host_src), n*sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice));
    }
    void memcpy_device_to_host(pointer host_dst, pointer device_src, size_type n){
        cuda_error_check(cudaMemcpy(static_cast<void*>(host_dst), static_cast<void*>(device_src.get()), n*sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    }
    void memcpy_device_to_device(pointer device_dst, pointer device_src, size_type n){
        cuda_error_check(cudaMemcpy(static_cast<void*>(device_dst.get()), static_cast<void*>(device_src.get()), n*sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToDevice));
    }
};

}   //end of namespace cuda_experimental

#endif