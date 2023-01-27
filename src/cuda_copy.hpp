#ifndef CUDA_COPY_HPP_
#define CUDA_COPY_HPP_

#include <immintrin.h>
#include "thread_pool.hpp"
#include "bounded_pool.hpp"
#include "cuda_pointer.hpp"
#include "cuda_allocator.hpp"

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
inline constexpr std::size_t multithread_threshold = 0;
//inline constexpr std::size_t multithread_threshold = 4*1024*1024;


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
    // template<std::size_t A>
    // auto align()const{
    //     static_assert(A%alignof(value_type) == 0);
    //     auto aligned_ = align<A>(begin_)
    //     return std::make_pair()
    // }
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
using locked_uninitialized_memory = cuda_uninitialized_memory<locked_allocator<std::byte>>;
using device_uninitialized_memory = cuda_uninitialized_memory<device_allocator<std::byte>>;

inline auto& locked_pool(){
    static bounded_pool::mc_bounded_pool<locked_uninitialized_memory> pool{locked_pool_size, locked_buffer_size};
    return pool;
}

inline auto& memcpy_workers_pool(){
    static thread_pool::thread_pool_v4 memcpy_pool{memcpy_pool_size};
    return memcpy_pool;
}
// inline auto& memcpy_workers_pool(){
//     static thread_pool::thread_pool_v1<void*(void*,const void*,std::size_t)> memcpy_pool{memcpy_pool_size, memcpy_pool_size};
//     return memcpy_pool;
// }

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
        auto n_chunk = n/N;
        auto n_last_chunk = n_chunk + n%N;
        auto dst_ = reinterpret_cast<unsigned char*>(dst);
        auto src_ = reinterpret_cast<const unsigned char*>(src);
        using future_type = decltype(memcpy_workers_pool().push(impl,dst_,src_,n_chunk));
        std::array<future_type, N-1> futures{};
        if (n_chunk){
            for (std::size_t i{0}; i!=N-1; ++i,dst_+=n_chunk,src_+=n_chunk){
                futures[i] = memcpy_workers_pool().push(impl, dst_,src_,n_chunk);
            }
        }
        impl(dst_,src_,n_last_chunk);
    }
    return dst;
}

//multithread uninitialized_copyn
//sync wrt caller thread, utilizes N threads: caller thread +  N-1 workers from pool
template<std::size_t N, typename It, typename Size, typename DIt>
auto uninitialized_copyn_multithread(It first, Size n, DIt d_first){
    static_assert(N>0);
    if (n){
        if (N>1){
            auto n_chunk = n/N;
            auto n_last_chunk = n_chunk + n%N;
            using impl_type = decltype(&std::uninitialized_copy_n<It,Size,DIt>);
            auto impl = static_cast<impl_type>(std::uninitialized_copy_n);
            using future_type = decltype(memcpy_workers_pool().push(impl,first,n_chunk,d_first));
            std::array<future_type, N-1> futures{};
            if (n_chunk){
                for (std::size_t i{0}; i!=N-1; ++i,first+=n_chunk,d_first+=n_chunk){
                    futures[i] = memcpy_workers_pool().push(impl, first,n_chunk,d_first);
                }
            }
            return impl(first, n_last_chunk, d_first);
        }else{
            return std::uninitialized_copy_n(first,n,d_first);
        }
    }else{
        return d_first;
    }
}


//dma transfer from locked buffer to device
inline auto dma_to_device(void* dst_device, decltype(locked_pool().pop()) src_locked, std::size_t n){
    cuda_error_check(cudaMemcpyAsync(dst_device, src_locked.get().data(), n, cudaMemcpyKind::cudaMemcpyHostToDevice, cuda_stream{}));
}
//memcpy multithread from locked buffer to pageable
inline auto memcpy_to_pageable(void* dst_pageable, decltype(locked_pool().pop()) src_locked, std::size_t n){
    memcpy_multithread<memcpy_workers>(dst_pageable, src_locked.get().data(), n, memcpy_impl);
}
//copy multithread from locked buffer to pageable It
template<typename It>
inline auto copy_to_pageable(It d_first, decltype(locked_pool().pop()) src_locked, std::size_t n){
    using value_type = typename std::iterator_traits<It>::value_type;
    uninitialized_copyn_multithread<memcpy_workers>(reinterpret_cast<value_type*>(src_locked.get().data().get()), n, d_first);
}

inline auto& async_pool(){
    return memcpy_workers_pool();
}
// inline auto& async_pool(){
//     static thread_pool::thread_pool_v4 async_pool_{async_pool_size};
//     return async_pool_;
// }
// inline auto& async_pool(){
//     static thread_pool::thread_pool_v1<decltype(dma_to_device)> async_pool_{async_pool_size, async_pool_size};
//     return async_pool_;
// }

}   //end of namespace cuda_memcpy

//pageable to device copy
//T must be trivially copyable
template<typename T>
void copy(T* first, T* last, device_pointer<std::remove_const_t<T>> d_first){
    std::cout<<std::endl<<"void copy(T* first, T* last, device_pointer<std::remove_const_t<T>> d_first){";
    auto n = std::distance(first,last)*sizeof(T);
    if (n>cuda_memcpy::multithread_threshold){
        auto n_chunks = n/cuda_memcpy::locked_buffer_size;
        auto last_chunk_size = n%cuda_memcpy::locked_buffer_size;
        auto src = reinterpret_cast<const std::byte*>(first);
        auto dst = reinterpret_cast<std::byte*>(d_first.get());
        using future_type = decltype(cuda_memcpy::async_pool().push(cuda_memcpy::dma_to_device, dst, cuda_memcpy::locked_pool().pop(), cuda_memcpy::locked_buffer_size));
        future_type async_future{};
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
    }else{
        cuda_error_check(cudaMemcpyAsync(d_first, first, n, cudaMemcpyKind::cudaMemcpyHostToDevice, cuda_stream{}));
    }
}

template<typename It, typename T, std::enable_if_t<!is_basic_pointer_v<It>, int> =0>
void copy(It first, It last, device_pointer<T> d_first){
    std::cout<<std::endl<<"void copy(It first, It last, device_pointer<typename std::iterator_traits<It>::value_type> d_first){";
    static_assert(std::is_same_v<std::decay_t<T>,typename std::iterator_traits<It>::value_type>);
    using value_type = typename std::iterator_traits<It>::value_type;
    auto n = std::distance(first,last);
    auto n_buffer = cuda_memcpy::locked_buffer_size/sizeof(value_type);
    auto n_buffer_bytes = n_buffer*sizeof(value_type);
    if (n>cuda_memcpy::multithread_threshold){
        auto n_chunks = n/n_buffer;
        auto last_chunk_size = n%n_buffer;
        using future_type = decltype(cuda_memcpy::async_pool().push(cuda_memcpy::dma_to_device, d_first, cuda_memcpy::locked_pool().pop(), cuda_memcpy::locked_buffer_size));
        future_type async_future{};
        for (std::size_t i{0}; i!=n_chunks; ++i,first+=n_buffer,d_first+=n_buffer){
            auto buf = cuda_memcpy::locked_pool().pop();
            //page-locked memory has 4096 alignment on most systems
            if (alignment(buf.get().data())%alignof(value_type)){
                //exception
            }
            auto buf_ = reinterpret_cast<value_type*>(buf.get().data().get());
            cuda_memcpy::uninitialized_copyn_multithread<cuda_memcpy::memcpy_workers>(first,n_buffer,buf_);  //sync copy pageable to locked
            if (async_future){async_future.wait();}
            async_future = cuda_memcpy::async_pool().push_async(cuda_memcpy::dma_to_device, d_first, buf, n_buffer_bytes);   //async dma transfer locked to device
        }
        if (last_chunk_size){
            auto buf = cuda_memcpy::locked_pool().pop();
            auto buf_ = reinterpret_cast<value_type*>(buf.get().data().get());
            cuda_memcpy::uninitialized_copyn_multithread<cuda_memcpy::memcpy_workers>(first,last_chunk_size,buf_);  //sync copy pageable to locked
            if (async_future){async_future.wait();}
            cuda_memcpy::async_pool().push(cuda_memcpy::dma_to_device, d_first, buf, last_chunk_size*sizeof(value_type));
        }
        if (async_future){async_future.wait();}
    }else{
        //cuda_error_check(cudaMemcpyAsync(d_first, first, n, cudaMemcpyKind::cudaMemcpyHostToDevice, cuda_stream{}));
    }
}

//device to pageable copy
template<typename T>
void copy(device_pointer<T> first, device_pointer<T> last, std::remove_const_t<T>* d_first){
    std::cout<<std::endl<<"void copy(device_pointer<T> first, device_pointer<T> last, std::remove_const_t<T>* d_first){";
    auto n = std::distance(first,last)*sizeof(T);
    if (n>cuda_memcpy::multithread_threshold){
        auto n_chunks = n/cuda_memcpy::locked_buffer_size;
        auto last_chunk_size = n%cuda_memcpy::locked_buffer_size;
        auto src = reinterpret_cast<const std::byte*>(first.get());
        auto dst = reinterpret_cast<std::byte*>(d_first);
        using future_type = decltype(cuda_memcpy::async_pool().push(cuda_memcpy::memcpy_to_pageable, dst, cuda_memcpy::locked_pool().pop(), cuda_memcpy::locked_buffer_size));
        future_type async_future{};
        for (std::size_t i{0}; i!=n_chunks; ++i,src+=cuda_memcpy::locked_buffer_size,dst+=cuda_memcpy::locked_buffer_size){
            auto buf = cuda_memcpy::locked_pool().pop();
            cuda_error_check(cudaMemcpyAsync(buf.get().data(), src, cuda_memcpy::locked_buffer_size, cudaMemcpyKind::cudaMemcpyDeviceToHost, cuda_stream{}));   //sync copy device to locked
            if (async_future){async_future.wait();}
            async_future = cuda_memcpy::async_pool().push_async(cuda_memcpy::memcpy_to_pageable, dst, buf, cuda_memcpy::locked_buffer_size);   //async copy locked to pageable
        }
        if (last_chunk_size){
            auto buf = cuda_memcpy::locked_pool().pop();
            cuda_error_check(cudaMemcpyAsync(buf.get().data(), src, last_chunk_size, cudaMemcpyKind::cudaMemcpyDeviceToHost, cuda_stream{}));   //sync from host to locked
            if (async_future){async_future.wait();}
            cuda_memcpy::async_pool().push(cuda_memcpy::memcpy_to_pageable, dst, buf, last_chunk_size);   //async copy from locked to pageable
        }
        if (async_future){async_future.wait();}
    }else{
        cuda_error_check(cudaMemcpyAsync(d_first, first, n, cudaMemcpyKind::cudaMemcpyDeviceToHost, cuda_stream{}));
    }
}

template<typename T, typename It, std::enable_if_t<!is_basic_pointer_v<It>,int> =0>
void copy(device_pointer<T> first, device_pointer<T> last, It d_first){
    std::cout<<std::endl<<"void copy(device_pointer<T> first, device_pointer<T> last, It d_first){";
    static_assert(std::is_same_v<std::decay_t<T>,typename std::iterator_traits<It>::value_type>);
    auto n = std::distance(first,last);
    auto n_buffer = cuda_memcpy::locked_buffer_size/sizeof(T);
    auto n_buffer_bytes = n_buffer*sizeof(T);
    if (n>cuda_memcpy::multithread_threshold){
        auto n_chunks = n/n_buffer;
        auto last_chunk_size = n%n_buffer;
        auto copy_to_pageable = static_cast<decltype(&cuda_memcpy::copy_to_pageable<It>)>(cuda_memcpy::copy_to_pageable);
        using future_type = decltype(cuda_memcpy::async_pool().push(copy_to_pageable, d_first, cuda_memcpy::locked_pool().pop(), cuda_memcpy::locked_buffer_size));
        future_type async_future{};
        for (std::size_t i{0}; i!=n_chunks; ++i,first+=n_buffer,d_first+=n_buffer){
            auto buf = cuda_memcpy::locked_pool().pop();
            cuda_error_check(cudaMemcpyAsync(buf.get().data(), first, n_buffer_bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost, cuda_stream{}));   //sync copy device to locked
            if (async_future){async_future.wait();}
            async_future = cuda_memcpy::async_pool().push_async(copy_to_pageable, d_first, buf, n_buffer);   //async copy locked to pageable
        }
        if (last_chunk_size){
            auto buf = cuda_memcpy::locked_pool().pop();
            cuda_error_check(cudaMemcpyAsync(buf.get().data(), first, last_chunk_size*sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost, cuda_stream{}));   //sync copy device to locked
            if (async_future){async_future.wait();}
            async_future = cuda_memcpy::async_pool().push_async(copy_to_pageable, d_first, buf, last_chunk_size);   //async copy locked to pageable
        }
        if (async_future){async_future.wait();}
    }else{
        //cuda_error_check(cudaMemcpyAsync(d_first, first, n, cudaMemcpyKind::cudaMemcpyHostToDevice, cuda_stream{}));
    }
}



}   //end of namespace cuda_experimental

#endif