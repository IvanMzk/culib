#ifndef CUDA_COPY_HPP_
#define CUDA_COPY_HPP_

#include <immintrin.h>
#include "cuda_pointer.hpp"
#include "cuda_allocator.hpp"
#include "thread_pool.hpp"
#include "bounded_pool.hpp"

namespace cuda_experimental{
namespace cuda_memcpy{

//host memcpy using avx
using avx_block_type = __m256i;
void* memcpy_avx(void* dst_host, const void* src_host, std::size_t n);
inline constexpr void*(*memcpy_impl)(void*,const void*,std::size_t) = std::memcpy;
//inline constexpr void*(*memcpy_impl)(void*,const void*,std::size_t) = memcpy_avx;

inline constexpr std::size_t memcpy_pool_size = 8;
inline constexpr std::size_t memcpy_workers = 4;
inline constexpr std::size_t async_pool_size = 2;
inline constexpr std::size_t locked_buffer_size = 64*1024*1024;
inline constexpr std::size_t locked_pool_size = 8;

template<typename Alloc>
class cuda_uninitialized_memory
{
public:
    using allocator_type = Alloc;
    using value_type = typename allocator_type::value_type;
    using pointer = typename allocator_type::pointer;
    using size_type = typename allocator_type::size_type;
    cuda_uninitialized_memory(const cuda_uninitialized_memory&) = delete;
    cuda_uninitialized_memory& operator=(const cuda_uninitialized_memory&) = delete;
    cuda_uninitialized_memory& operator=(cuda_uninitialized_memory&&) = delete;
    ~cuda_uninitialized_memory(){deallocate();}
    explicit cuda_uninitialized_memory(const size_type& n, const allocator_type& alloc = allocator_type{}):
        allocator_{alloc},
        size_{n},
        begin_{allocate(n)}
    {}
    cuda_uninitialized_memory(cuda_uninitialized_memory&& other):
        allocator_{std::move(other.allocator_)},
        size_{other.size_},
        begin_{other.begin_}
    {
        other.size_ = 0;
        other.begin_ = nullptr;
    }
    auto data()const{return begin_;}
    auto size()const{return size_;}
private:
    pointer allocate(const size_type& n){
        return allocator_.allocate(n);
    }
    void deallocate(){
        if (begin_){
            allocator_.deallocate(begin_,size_);
            size_ = 0;
            begin_ = nullptr;
        }
    }
    allocator_type allocator_;
    size_type size_;
    pointer begin_;
};
using locked_uninitialized_memory = cuda_uninitialized_memory<locked_allocator<unsigned char>>;
using device_uninitialized_memory = cuda_uninitialized_memory<device_allocator<unsigned char>>;

inline auto& locked_pool(){
    static bounded_pool::mc_bounded_pool<locked_uninitialized_memory> pool{locked_pool_size, locked_buffer_size};
    return pool;
}
inline auto& memcpy_workers_pool(){
    static thread_pool::thread_pool_v1<void*(void*,const void*,std::size_t)> memcpy_pool{memcpy_pool_size, memcpy_pool_size};
    return memcpy_pool;
}

//multithread memcpy
//sync wrt caller thread, utilizes N threads: caller thread +  N-1 workers from pool
template<std::size_t>
void* memcpy_multithread(void*, const void*, std::size_t, void*(*)(void*,const void*,std::size_t));
template<>
inline void* memcpy_multithread<1>(void* dst, const void* src, std::size_t n, void*(*impl)(void*,const void*,std::size_t)){
    return impl(dst,src,n);
}
template<std::size_t N>
void* memcpy_multithread(void* dst, const void* src, std::size_t n, void*(*impl)(void*,const void*,std::size_t)){
    static_assert(N>1);
    if (n!=0){
        std::array<std::remove_reference_t<decltype(memcpy_workers_pool())>::future_type, N-1> futures{};
        auto n_chunk = n/N;
        auto n_last_chunk = n_chunk + n%N;
        auto dst_ = reinterpret_cast<unsigned char*>(dst);
        auto src_ = reinterpret_cast<const unsigned char*>(src);
        if (n_chunk){
            for (std::size_t i{0}; i!=N-1; ++i,dst_+=n_chunk,src_+=n_chunk){
                futures[i] = memcpy_workers_pool().push(impl, dst_,src_,n_chunk);
            }
        }
        impl(dst_,src_,n_last_chunk);
    }
    return dst;
}

//return aligned pointer that is nearest to p and greater or equal to p
//A - required alignment in bytes
template<std::size_t A>
inline auto align(const void* p){
    static_assert(A != 0);
    static_assert((A&(A-1))  == 0);
    return reinterpret_cast<const void *>((reinterpret_cast<std::uintptr_t>(p)+(A-1)) & ~(A-1));
}
template<std::size_t A>
inline auto align(void* p){
    return const_cast<void*>(align<A>(const_cast<const void*>(p)));
}

//dma transfer from locked buffer to device
inline auto dma_to_device(void* dst_device, decltype(locked_pool().pop()) src_locked, std::size_t n){
    cuda_error_check(cudaMemcpyAsync(dst_device, src_locked.get().data(), n, cudaMemcpyKind::cudaMemcpyHostToDevice, cuda_stream{}));
}
//memcpy multithread from locked buffer to pageable
inline auto copy_to_pageable(void* dst_pageable, decltype(locked_pool().pop()) src_locked, std::size_t n){
    memcpy_multithread<memcpy_workers>(dst_pageable, src_locked.get().data(), n, memcpy_impl);
}
inline auto& async_pool(){
    static thread_pool::thread_pool_v1<decltype(dma_to_device)> async_pool_{async_pool_size, async_pool_size};
    return async_pool_;
}

}   //end of namespace cuda_memcpy

//pageable to device copy
template<typename T>
void copy(const T* first, const T* last, device_pointer<T> d_first){
    auto n = std::distance(first,last)*sizeof(T);
    auto n_chunks = n/cuda_memcpy::locked_buffer_size;
    auto last_chunk_size = n%cuda_memcpy::locked_buffer_size;
    auto src = reinterpret_cast<const unsigned char*>(first);
    auto dst = static_cast<device_pointer<unsigned char>>(d_first);
    typename std::remove_reference_t<decltype(cuda_memcpy::async_pool())>::future_type async_future{};
    for (std::size_t i{0}; i!=n_chunks; ++i,src+=cuda_memcpy::locked_buffer_size,dst+=cuda_memcpy::locked_buffer_size){
        auto buf = cuda_memcpy::locked_pool().pop();
        cuda_memcpy::memcpy_multithread<cuda_memcpy::memcpy_workers>(buf.get().data(), src, cuda_memcpy::locked_buffer_size, cuda_memcpy::memcpy_impl);   //sync copy pageable to locked
        if (async_future){async_future.wait();}
        async_future = cuda_memcpy::async_pool().push_async(cuda_memcpy::dma_to_device, dst, buf, cuda_memcpy::locked_buffer_size);   //async dma transfer locked to device
    }
    if (last_chunk_size){
        auto buf = cuda_memcpy::locked_pool().pop();
        cuda_memcpy::memcpy_multithread<cuda_memcpy::memcpy_workers>(buf.get().data(), src, last_chunk_size, cuda_memcpy::memcpy_impl);
        if (async_future){async_future.wait();}
        cuda_memcpy::async_pool().push(cuda_memcpy::dma_to_device, dst, buf, last_chunk_size);
    }
    if (async_future){async_future.wait();}
}

//device to pageable copy
template<typename T>
void copy(device_pointer<T> first, device_pointer<T> last, std::remove_const_t<T>* d_first){
    auto n = std::distance(first,last)*sizeof(T);
    auto n_chunks = n/cuda_memcpy::locked_buffer_size;
    auto last_chunk_size = n%cuda_memcpy::locked_buffer_size;
    auto src = static_cast<device_pointer<const unsigned char>>(first);
    auto dst = reinterpret_cast<unsigned char*>(d_first);
    typename std::remove_reference_t<decltype(cuda_memcpy::async_pool())>::future_type async_future{};
    for (std::size_t i{0}; i!=n_chunks; ++i,src+=cuda_memcpy::locked_buffer_size,dst+=cuda_memcpy::locked_buffer_size){
        auto buf = cuda_memcpy::locked_pool().pop();
        cuda_error_check(cudaMemcpyAsync(buf.get().data(), src, cuda_memcpy::locked_buffer_size, cudaMemcpyKind::cudaMemcpyDeviceToHost, cuda_stream{}));   //sync copy device to locked
        if (async_future){async_future.wait();}
        async_future = cuda_memcpy::async_pool().push_async(cuda_memcpy::copy_to_pageable, dst, buf, cuda_memcpy::locked_buffer_size);   //async copy locked to pageable
    }
    if (last_chunk_size){
        auto buf = cuda_memcpy::locked_pool().pop();
        cuda_error_check(cudaMemcpyAsync(buf.get().data(), src, last_chunk_size, cudaMemcpyKind::cudaMemcpyDeviceToHost, cuda_stream{}));   //sync from host to locked
        if (async_future){async_future.wait();}
        cuda_memcpy::async_pool().push(cuda_memcpy::copy_to_pageable, dst, buf, last_chunk_size);   //async copy from locked to pageable
    }
    if (async_future){async_future.wait();}
}


}   //end of namespace cuda_experimental

#endif