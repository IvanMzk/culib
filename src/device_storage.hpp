#ifndef DEVICE_STORAGE_HPP_
#define DEVICE_STORAGE_HPP_

#include <memory>
#include "cuda_runtime.h"
#include "cuda_helpers.hpp"

namespace cuda_experimental{

namespace detail{
    template<typename, typename = void> constexpr bool is_iterator = false;
    template<typename T> constexpr bool is_iterator<T,std::void_t<typename std::iterator_traits<T>::iterator_category>> = true;
}   //end of namespace detail


template<typename T>
class cuda_allocator
{
public:
    using difference_type = std::ptrdiff_t;
    using size_type = difference_type;
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

template<typename T>
class cuda_memory_handler : public cuda_allocator<T>
{
public:
    using typename cuda_allocator::size_type;
    using typename cuda_allocator::value_type;
    using typename cuda_allocator::pointer;

    void memcpy_host_to_device(pointer device_dst, pointer host_src, size_type n){
        cuda_error_check(cudaMemcpy(static_cast<void*>(device_dst), static_cast<void*>(host_src), n*sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice));
    }
    void memcpy_device_to_host(pointer host_dst, pointer device_src, size_type n){
        cuda_error_check(cudaMemcpy(static_cast<void*>(host_dst), static_cast<void*>(device_src), n*sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    }
    void memcpy_device_to_device(pointer device_dst, pointer device_src, size_type n){
        cuda_error_check(cudaMemcpy(static_cast<void*>(device_dst), static_cast<void*>(device_src), n*sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToDevice));
    }
};

template<typename T, typename DiffT>
class device_pointer{
    using difference_type = DiffT;
    using value_type = T;
    using pointer = T*;

    pointer ptr;
public:
    device_pointer() = default;
    explicit device_pointer(pointer ptr_):
        ptr{ptr_}
    {}
    operator bool()const{return static_cast<bool>(ptr);}
    auto get()const{return ptr;}
};

template<typename T, typename DiffT>
auto operator==(const device_pointer<T,DiffT>& lhs, const device_pointer<T,DiffT>& rhs){return lhs.get() == rhs.get();}
template<typename T, typename DiffT>
auto operator!=(const device_pointer<T,DiffT>& lhs, const device_pointer<T,DiffT>& rhs){return !(lhs != rhs);}
template<typename T, typename DiffT>
auto operator-(const device_pointer<T,DiffT>& lhs, const device_pointer<T,DiffT>& rhs){return lhs.get() - rhs.get();}
template<typename T, typename DiffT>
auto operator+(const device_pointer<T,DiffT>& lhs, const DiffT& rhs){return device_pointer<T,DiffT>{lhs.get() + rhs};}
template<typename T, typename DiffT>
auto operator+(const DiffT& lhs, const device_pointer<T,DiffT>& rhs){return rhs+lhs;}



template<typename T, typename MemoryHandler = cuda_memory_handler<T>>
class device_storage : private MemoryHandler
{
    using memory_handler_type = MemoryHandler;
public:
    using difference_type = typename memory_handler_type::difference_type;
    using size_type = typename memory_handler_type::size_type;
    using value_type = typename memory_handler_type::value_type;
    using pointer = typename memory_handler_type::pointer;
    using device_pointer_type = typename device_pointer<value_type,difference_type>;

    ~device_storage(){deallocate();}
    device_storage& operator=(const device_storage&) = delete;
    device_storage& operator=(device_storage&&) = delete;
    device_storage() = default;
    device_storage(device_storage&& other):
        size_{other.size_},
        device_begin_{other.device_begin_}
    {
        other.size_ = 0;
        other.device_begin_ = nullptr;
    }
    explicit device_storage(const size_type& n):
        size_{n},
        device_begin_{allocate(n)}
    {}
    device_storage(const size_type& n, const value_type& v):
        size_{n},
        device_begin_{allocate(n)}
    {
        auto buffer = std::make_unique<value_type>(n);
        std::uninitialized_fill_n(buffer.get(),n,v);
        memcpy_host_to_device(device_begin_,buffer.get(),n);
    }
    template<typename It, std::enable_if_t<detail::is_iterator<It> ,int> =0 >
    device_storage(It host_begin, It host_end):
        size_{std::distance(host_begin,host_end)},
        device_begin_{allocate(size_)}
    {
        auto buffer = std::make_unique<value_type>(size_);
        std::uninitialized_copy(host_begin,host_end,buffer.get());
        memcpy_host_to_device(device_begin_,buffer.get(),size_);
    }
    template<typename U, std::enable_if_t<std::is_convertible_v<U,value_type>,int> =0 >
    device_storage(std::initializer_list<U> init_data):
        size_{init_data.size()},
        device_begin_{allocate(size_)}
    {
        memcpy_host_to_device(device_begin_,init_data.begin(),size_);
    }

    auto device_begin()const{return device_pointer_type{device_begin_};}
    auto device_end()const{return  device_pointer_type{device_begin_ + size_};}
    auto size()const{return size_;}
    void free()const{deallocate();}
    auto clone()const{return device_storage{*this};}

private:
    device_storage(const device_storage& other):
        size_{other.size_},
        device_begin_{allocate(other.size_)}
    {
        memcpy_device_to_device(device_begin_,other.device_begin_,size_)
    }

    pointer allocate(const size_type& n){
        return memory_handler_type::allocate(n);
    }
    void deallocate(){
        if (device_begin_){
            memory_handler_type::deallocate(device_begin_,size_);
            size_ = 0;
            device_begin_ = nullptr;
        }
    }

    size_type size_;
    pointer device_begin_;
};

}   //end of namespace cuda_experimental



#endif