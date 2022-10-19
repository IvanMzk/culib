#ifndef CUDA_MEMORY_HPP_
#define CUDA_MEMORY_HPP_

#include <iterator>
#include <memory>
#include "cuda_helpers.hpp"

namespace cuda_experimental{

template<typename T>
class cuda_pointer{
public:
    using value_type = T;
    using pointer = T*;

    cuda_pointer() = default;
    cuda_pointer(const cuda_pointer&) = default;
    cuda_pointer& operator=(const cuda_pointer&) = default;
    explicit cuda_pointer(pointer ptr_):
        ptr{ptr_}
    {}
    cuda_pointer& operator=(std::nullptr_t){
        ptr = nullptr;
        return *this;
    }

    operator cuda_pointer<const T>()const{return cuda_pointer<const T>{ptr};}
    operator bool()const{return static_cast<bool>(ptr);}
    pointer get()const{return ptr;}
private:
    pointer ptr;
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
auto ptr_to_void(const cuda_pointer<T>& p){return static_cast<std::conditional_t<std::is_const_v<T>,const void*,void*>>(p.get());}
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
    using const_pointer = cuda_pointer<const T>;

    pointer allocate(size_type n){
        void* p;
        cuda_error_check(cudaMalloc(&p,n*sizeof(T)));
        return pointer{reinterpret_cast<T*>(p)};
    }
    void deallocate(pointer p, size_type n){
        cuda_error_check(cudaFree(ptr_to_void(p)));
    }
};

template<typename T, typename SizeT>
auto make_host_buffer(const SizeT& n){
    return std::make_unique<T[]>(n);
}

//copy from host to device
template<typename T>
void copy(const T* first, const T* last, cuda_pointer<T> d_first){
    auto n = std::distance(first,last);
    cuda_error_check(cudaMemcpy(ptr_to_void(d_first), ptr_to_void(first), n*sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice));
}
template<typename It, std::enable_if_t<!std::is_pointer_v<It>,int> =0 >
void copy(It first, It last, cuda_pointer<typename std::iterator_traits<It>::value_type> d_first){
    static_assert(!std::is_pointer_v<It>);
    auto n = std::distance(first,last);
    auto buffer = make_host_buffer<std::iterator_traits<It>::value_type>(n);
    std::uninitialized_copy_n(first,n,buffer.get());
    copy(buffer.get(),buffer.get()+n,d_first);
}
//copy from device to host
template<typename T>
void copy(cuda_pointer<T> first, cuda_pointer<T> last, std::remove_const_t<T>* d_first){
    auto n = distance(first,last);
    cuda_error_check(cudaMemcpy(ptr_to_void(d_first), ptr_to_void(first), n*sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost));
}
template<typename T, typename It, std::enable_if_t<!std::is_pointer_v<It>,int> =0>
void copy(cuda_pointer<T> first, cuda_pointer<T> last, It d_first){
    static_assert(!std::is_pointer_v<It>);
    static_assert(std::is_same_v<std::decay_t<T>, typename std::iterator_traits<It>::value_type>);
    auto n = distance(first,last);
    auto buffer = make_host_buffer<std::iterator_traits<It>::value_type>(n);
    copy(first,last,buffer.get());
    std::copy_n(buffer.get(),n,d_first);
}
//copy from device range to device
template<typename T>
void copy(cuda_pointer<T> first, cuda_pointer<T> last, cuda_pointer<std::remove_const_t<T>> d_first){
    auto n = distance(first,last);
    cuda_error_check(cudaMemcpy(ptr_to_void(d_first), ptr_to_void(first), n*sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToDevice));
}

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