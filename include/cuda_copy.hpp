/*
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#ifndef CUDA_COPY_HPP_
#define CUDA_COPY_HPP_

#include <exception>
#include <cstring>
#include <immintrin.h>
#include "multithreading.hpp"
#include "cuda_pointer.hpp"
#include "cuda_allocator.hpp"
#include "cuda_copy_config.hpp"

namespace culib{
namespace cuda_copy{

class bad_alignment_exception : public std::runtime_error
{
public:
    bad_alignment_exception():
        bad_alignment_exception{""}
    {}
    bad_alignment_exception(const char* what):
        std::runtime_error{what}
    {}
};

using avx_block_type = __m256i;
void* memcpy_avx(void* dst_host, const void* src_host, std::size_t n);
inline constexpr void*(*memcpy_impl)(void*,const void*,std::size_t) = [](){if constexpr (native_memcpy){return std::memcpy;}else{return memcpy_avx;}}();

template<typename Alloc, std::size_t Alignment = 0>
class cuda_uninitialized_memory
{
public:
    static_assert(Alignment==0 || (Alignment&(Alignment-1))==0);    //must be zero or pow of 2
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
    {
        if constexpr (Alignment != 0){
            if (alignment(begin_)%Alignment != 0){ //allocated memory not aligned as required
                deallocate();
                throw bad_alignment_exception{};
            }
        }
    }
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
using locked_uninitialized_memory = cuda_uninitialized_memory<locked_allocator<std::byte>,locked_buffer_alignment>;

inline auto& locked_pool(){
    static culib::multithreading::mc_bounded_pool<locked_uninitialized_memory> pool{locked_pool_size, locked_buffer_size};
    return pool;
}
inline auto& copy_pool(){
    return culib::multithreading::get_pool();
}

//multithread memcpy
//sync wrt caller thread, utilizes N threads: caller thread +  N-1 workers from pool
template<std::size_t N>
void* memcpy_multithread(void* dst, const void* src, std::size_t n, void*(*impl)(void*,const void*,std::size_t)){
    static_assert(N>0);
    if (n!=0){
        if constexpr (N>1){
            auto n_chunk = n/N;
            auto n_last_chunk = n_chunk + n%N;
            auto dst_ = reinterpret_cast<unsigned char*>(dst);
            auto src_ = reinterpret_cast<const unsigned char*>(src);
            using future_type = decltype(copy_pool().push(impl,dst_,src_,n_chunk));
            std::array<future_type, N-1> futures{};
            if (n_chunk){
                for (std::size_t i{0}; i!=N-1; ++i,dst_+=n_chunk,src_+=n_chunk){
                    futures[i] = copy_pool().push(impl, dst_,src_,n_chunk);
                }
            }
            impl(dst_,src_,n_last_chunk);
        }else{
            impl(dst,src,n);
        }
    }
    return dst;
}

template<typename It>
constexpr bool is_random_access_iterator_v = std::is_convertible_v<typename std::iterator_traits<It>::iterator_category, std::random_access_iterator_tag>;

//like library version but returns both iterators updated
template<typename InputIt, typename Size, typename ForwardIt>
inline auto uninitialized_copyn_(InputIt first, Size n, ForwardIt d_first, std::input_iterator_tag){
    using value_type = typename std::iterator_traits<ForwardIt>::value_type;
    for (; n > 0; ++first, (void) ++d_first, --n) {
        ::new (const_cast<void*>(static_cast<const volatile void*>(std::addressof(*d_first)))) value_type(*first);
    }
    return std::pair<InputIt,ForwardIt>{first,d_first};
}
template<typename InputIt, typename Size, typename ForwardIt>
inline auto uninitialized_copyn_(InputIt first, Size n, ForwardIt d_first, std::random_access_iterator_tag){
    return std::pair<InputIt,ForwardIt>{std::next(first,n),std::uninitialized_copy_n(first,n,d_first)};
}
template<typename InputIt, typename Size, typename ForwardIt>
inline auto uninitialized_copyn(InputIt first, Size n, ForwardIt d_first){
    return uninitialized_copyn_(first,n,d_first,typename std::iterator_traits<InputIt>::iterator_category{});
}

template<typename InputIt, typename Size, typename ForwardIt>
inline auto copyn_(InputIt first, Size n, ForwardIt d_first, std::input_iterator_tag){
    for (; n > 0; ++first, (void) ++d_first, --n) {
        *d_first = *first;
    }
    return std::pair<InputIt,ForwardIt>{first,d_first};
}
template<typename InputIt, typename Size, typename ForwardIt>
inline auto copyn_(InputIt first, Size n, ForwardIt d_first, std::random_access_iterator_tag){
    return std::pair<InputIt,ForwardIt>{std::next(first,n),std::copy_n(first,n,d_first)};
}
template<typename InputIt, typename Size, typename ForwardIt>
inline auto copyn(InputIt first, Size n, ForwardIt d_first){
    return copyn_(first,n,d_first,typename std::iterator_traits<InputIt>::iterator_category{});
}

//multithread copyn
//sync wrt caller thread, utilizes N threads: caller thread +  N-1 workers from pool
template<std::size_t N, typename It, typename Size, typename DIt, typename Impl>
auto copyn_multithread_(It first, Size n, DIt d_first, Impl impl){
    static_assert(N>0);
    if (n>0){
        if constexpr (N>1 && is_random_access_iterator_v<It> && is_random_access_iterator_v<DIt>){
            auto n_chunk = n/N;
            auto n_last_chunk = n_chunk + n%N;
            using future_type = decltype(copy_pool().push(impl,first,n_chunk,d_first));
            std::array<future_type, N-1> futures{};
            if (n_chunk){
                for (std::size_t i{0}; i!=N-1; ++i,first+=n_chunk,d_first+=n_chunk){
                    futures[i] = copy_pool().push(impl, first,n_chunk,d_first);
                }
            }
            return impl(first, n_last_chunk, d_first);
        }else{  //copy in single thread if N == 1 or both iterators not at least random access
            return impl(first,n,d_first);
        }
    }else{
        return std::pair<It,DIt>{first, d_first};
    }
}

template<std::size_t N, typename It, typename Size, typename DIt>
inline auto uninitialized_copyn_multithread(It first, Size n, DIt d_first){
    return copyn_multithread_<N>(first, n, d_first, static_cast<decltype(&uninitialized_copyn<It,Size,DIt>)>(uninitialized_copyn));
}
template<std::size_t N, typename It, typename Size, typename DIt>
inline auto copyn_multithread(It first, Size n, DIt d_first){
    return copyn_multithread_<N>(first, n, d_first, static_cast<decltype(&copyn<It,Size,DIt>)>(copyn));
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
    static_assert(locked_buffer_alignment%alignof(value_type) == 0);
    return copyn_multithread<memcpy_workers>(reinterpret_cast<value_type*>(src_locked.get().data().get()), n, d_first);
}

template<typename T> struct copier;

template<>
struct copier<native_copier_tag>{
//pageable to device
template<typename T>
static auto copy(const T* first, const T* last, device_pointer<std::remove_const_t<T>> d_first){
    auto n = std::distance(first,last);
    cuda_error_check(cudaMemcpyAsync(d_first, first, n*sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice, cuda_stream{}));
    return d_first+n;
}

template<typename It, typename T, std::enable_if_t<!culib::detail::is_basic_pointer_v<It>, int> =0>
static auto copy(It first, It last, device_pointer<T> d_first){
    using value_type = typename std::iterator_traits<It>::value_type;
    static_assert(locked_buffer_alignment%alignof(value_type) == 0);
    static_assert(std::is_same_v<std::decay_t<T>,value_type>);
    auto n = static_cast<std::size_t>(std::distance(first, last));
    auto n_buffer = locked_buffer_size/sizeof(value_type);
    for(;n >= n_buffer; n-=n_buffer, d_first+=n_buffer ){
        auto buf = locked_pool().pop();
        auto buf_first = reinterpret_cast<value_type*>(buf.get().data().get());
        first = uninitialized_copyn(first, n_buffer, buf_first).first;
        cuda_error_check(cudaMemcpyAsync(d_first, buf_first, n_buffer*sizeof(value_type), cudaMemcpyKind::cudaMemcpyHostToDevice, cuda_stream{}));
    }
    if(n){
        auto buf = locked_pool().pop();
        auto buf_first = reinterpret_cast<value_type*>(buf.get().data().get());
        std::uninitialized_copy_n(first, n, buf_first);
        cuda_error_check(cudaMemcpyAsync(d_first, buf_first, n*sizeof(value_type), cudaMemcpyKind::cudaMemcpyHostToDevice, cuda_stream{}));
        d_first+=n;
    }
    return d_first;
}

//device to pageable
template<typename T>
static auto copy(device_pointer<T> first, device_pointer<T> last, std::remove_const_t<T>* d_first){
    auto n = std::distance(first,last);
    cuda_error_check(cudaMemcpyAsync(d_first, first, n*sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost, cuda_stream{}));
    return d_first+n;
}

template<typename T, typename It, std::enable_if_t<!culib::detail::is_basic_pointer_v<It>,int> =0>
static auto copy(device_pointer<T> first, device_pointer<T> last, It d_first){
    using value_type = typename std::iterator_traits<It>::value_type;
    static_assert(locked_buffer_alignment%alignof(value_type) == 0);
    static_assert(std::is_same_v<std::decay_t<T>,value_type>);
    auto n = static_cast<std::size_t>(std::distance(first, last));
    auto n_buffer = locked_buffer_size/sizeof(value_type);
    for(;n >= n_buffer; n-=n_buffer, first+=n_buffer){
        auto buf = locked_pool().pop();
        auto buf_first = reinterpret_cast<value_type*>(buf.get().data().get());
        cuda_error_check(cudaMemcpyAsync(buf_first, first, n_buffer*sizeof(value_type), cudaMemcpyKind::cudaMemcpyDeviceToHost, cuda_stream{})); //int[] cast to byte and copy to byte[] - ok
        d_first = std::copy_n(buf_first, n_buffer, d_first);
    }
    if(n){
        auto buf = locked_pool().pop();
        auto buf_first = reinterpret_cast<value_type*>(buf.get().data().get());
        cuda_error_check(cudaMemcpyAsync(buf_first, first, n*sizeof(value_type), cudaMemcpyKind::cudaMemcpyDeviceToHost, cuda_stream{}));
        d_first = std::copy_n(buf_first, n, d_first);
    }
    return d_first;
}

//device to device
//assume UVA
//to copy to peer without staging cudaDeviceEnablePeerAccess should be called beforehand
template<typename T>
static auto copy(device_pointer<T> first, device_pointer<T> last, device_pointer<std::remove_const_t<T>> d_first){
    auto n = std::distance(first,last);
    cuda_error_check(cudaMemcpyAsync(d_first, first, n*sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToDevice, cuda_stream{}));
    return d_first+n;
}

};  //end of struct copier<native_copier_tag>{

template<>
struct copier<multithread_copier_tag>{
//pageable to device copy
template<typename T>
static auto copy(T* first, T* last, device_pointer<std::remove_const_t<T>> d_first){
    auto n = std::distance(first,last);
    auto n_bytes = n*sizeof(T);
    if (n_bytes>cuda_copy::multithread_threshold){
        auto src = reinterpret_cast<const std::byte*>(first);
        auto dst = reinterpret_cast<std::byte*>(d_first.get());
        using future_type = decltype(cuda_copy::copy_pool().push(cuda_copy::dma_to_device, dst, cuda_copy::locked_pool().pop(), cuda_copy::locked_buffer_size));
        future_type async_future{};
        for (; n_bytes>=cuda_copy::locked_buffer_size; n_bytes-=cuda_copy::locked_buffer_size, src+=cuda_copy::locked_buffer_size,dst+=cuda_copy::locked_buffer_size){
            auto buf = cuda_copy::locked_pool().pop();
            cuda_copy::memcpy_multithread<cuda_copy::memcpy_workers>(buf.get().data(), src, cuda_copy::locked_buffer_size, cuda_copy::memcpy_impl);   //sync copy pageable to locked
            if (async_future){async_future.wait();}
            async_future = cuda_copy::copy_pool().push_async(cuda_copy::dma_to_device, dst, buf, cuda_copy::locked_buffer_size);   //async dma transfer locked to device
        }
        if (n_bytes){
            auto buf = cuda_copy::locked_pool().pop();
            cuda_copy::memcpy_multithread<cuda_copy::memcpy_workers>(buf.get().data(), src, n_bytes, cuda_copy::memcpy_impl);
            if (async_future){async_future.wait();}
            cuda_copy::copy_pool().push(cuda_copy::dma_to_device, dst, buf, n_bytes);
        }
        if (async_future){async_future.wait();}
        return d_first+n;
    }else{
        return copier<native_copier_tag>::copy(first,last,d_first);
    }
}

template<typename It, typename T, std::enable_if_t<!culib::detail::is_basic_pointer_v<It>, int> =0>
static auto copy(It first, It last, device_pointer<T> d_first){
    static_assert(std::is_same_v<std::decay_t<T>,typename std::iterator_traits<It>::value_type>);
    using value_type = typename std::iterator_traits<It>::value_type;
    static_assert(locked_buffer_alignment%alignof(value_type) == 0);
    auto n = static_cast<std::size_t>(std::distance(first,last));
    auto n_bytes = n*sizeof(value_type);
    auto n_buffer = locked_buffer_size/sizeof(value_type);
    auto n_buffer_bytes = n_buffer*sizeof(value_type);
    if (n_bytes>cuda_copy::multithread_threshold){
        using future_type = decltype(cuda_copy::copy_pool().push(cuda_copy::dma_to_device, d_first, cuda_copy::locked_pool().pop(), cuda_copy::locked_buffer_size));
        future_type async_future{};
        for (; n>=n_buffer; n-=n_buffer,d_first+=n_buffer){
            auto buf = cuda_copy::locked_pool().pop();
            first = cuda_copy::uninitialized_copyn_multithread<cuda_copy::memcpy_workers>(first,n_buffer,reinterpret_cast<value_type*>(buf.get().data().get())).first;  //sync copy pageable to locked
            if (async_future){async_future.wait();}
            async_future = cuda_copy::copy_pool().push_async(cuda_copy::dma_to_device, d_first, buf, n_buffer_bytes);   //async dma transfer locked to device
        }
        if (n){
            auto buf = cuda_copy::locked_pool().pop();
            cuda_copy::uninitialized_copyn_multithread<cuda_copy::memcpy_workers>(first,n,reinterpret_cast<value_type*>(buf.get().data().get()));  //sync copy pageable to locked
            if (async_future){async_future.wait();}
            cuda_copy::copy_pool().push(cuda_copy::dma_to_device, d_first, buf, n*sizeof(value_type));
            d_first+=n;
        }
        if (async_future){async_future.wait();}
        return d_first;
    }else{
        return copier<native_copier_tag>::copy(first,last,d_first);
    }
}

//device to pageable copy
template<typename T>
static auto copy(device_pointer<T> first, device_pointer<T> last, std::remove_const_t<T>* d_first){
    auto n = std::distance(first,last);
    auto n_bytes = n*sizeof(T);
    if (n_bytes>cuda_copy::multithread_threshold){
        auto src = reinterpret_cast<const std::byte*>(first.get());
        auto dst = reinterpret_cast<std::byte*>(d_first);
        using future_type = decltype(cuda_copy::copy_pool().push(cuda_copy::memcpy_to_pageable, dst, cuda_copy::locked_pool().pop(), cuda_copy::locked_buffer_size));
        future_type async_future{};
        for (; n_bytes>=cuda_copy::locked_buffer_size; n_bytes-=cuda_copy::locked_buffer_size, src+=cuda_copy::locked_buffer_size, dst+=cuda_copy::locked_buffer_size){
            auto buf = cuda_copy::locked_pool().pop();
            cuda_error_check(cudaMemcpyAsync(buf.get().data(), src, cuda_copy::locked_buffer_size, cudaMemcpyKind::cudaMemcpyDeviceToHost, cuda_stream{}));   //sync copy device to locked
            if (async_future){async_future.wait();}
            async_future = cuda_copy::copy_pool().push_async(cuda_copy::memcpy_to_pageable, dst, buf, cuda_copy::locked_buffer_size);   //async copy locked to pageable
        }
        if (n_bytes){
            auto buf = cuda_copy::locked_pool().pop();
            cuda_error_check(cudaMemcpyAsync(buf.get().data(), src, n_bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost, cuda_stream{}));   //sync from host to locked
            if (async_future){async_future.wait();}
            cuda_copy::copy_pool().push(cuda_copy::memcpy_to_pageable, dst, buf, n_bytes);   //async copy from locked to pageable
        }
        if (async_future){async_future.wait();}
        return d_first+n;
    }else{
        return copier<native_copier_tag>::copy(first,last,d_first);
    }
}

template<typename T, typename It, std::enable_if_t<!culib::detail::is_basic_pointer_v<It>,int> =0>
static auto copy(device_pointer<T> first, device_pointer<T> last, It d_first){
    static_assert(std::is_same_v<std::decay_t<T>,typename std::iterator_traits<It>::value_type>);
    auto n = static_cast<std::size_t>(std::distance(first,last));
    auto n_bytes = n*sizeof(T);
    auto n_buffer = cuda_copy::locked_buffer_size/sizeof(T);
    auto n_buffer_bytes = n_buffer*sizeof(T);
    if (n_bytes>cuda_copy::multithread_threshold){
        auto copy_to_pageable = static_cast<decltype(&cuda_copy::copy_to_pageable<It>)>(cuda_copy::copy_to_pageable);
        using future_type = decltype(cuda_copy::copy_pool().push(copy_to_pageable, d_first, cuda_copy::locked_pool().pop(), cuda_copy::locked_buffer_size));
        future_type async_future{};
        for (; n>=n_buffer; n-=n_buffer, first+=n_buffer){
            auto buf = cuda_copy::locked_pool().pop();
            cuda_error_check(cudaMemcpyAsync(buf.get().data(), first, n_buffer_bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost, cuda_stream{}));   //sync copy device to locked
            if (async_future){
                d_first = async_future.get().second;
            }
            async_future = cuda_copy::copy_pool().push_async(copy_to_pageable, d_first, buf, n_buffer);   //async copy locked to pageable
        }
        if (n){
            auto buf = cuda_copy::locked_pool().pop();
            cuda_error_check(cudaMemcpyAsync(buf.get().data(), first, n*sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost, cuda_stream{}));   //sync copy device to locked
            if (async_future){
                d_first = async_future.get().second;
            }
            async_future = cuda_copy::copy_pool().push_async(copy_to_pageable, d_first, buf, n);   //async copy locked to pageable
        }
        if (async_future){
                d_first = async_future.get().second;
        }
        return d_first;
    }else{
        return copier<native_copier_tag>::copy(first,last,d_first);
    }
}

//device device copy
template<typename T>
static auto copy(device_pointer<T> first, device_pointer<T> last, device_pointer<std::remove_const_t<T>> d_first){
    auto n = std::distance(first,last);
    auto n_bytes = n*sizeof(T);
    if (n_bytes>cuda_copy::peer_multithread_threshold && first.device()!=d_first.device()){
        auto src = reinterpret_cast<const std::byte*>(first.get());
        auto dst = reinterpret_cast<std::byte*>(d_first.get());
        using future_type = decltype(cuda_copy::copy_pool().push(cuda_copy::dma_to_device, dst, cuda_copy::locked_pool().pop(), cuda_copy::locked_buffer_size));
        future_type async_future{};
        for (; n_bytes>=cuda_copy::locked_buffer_size; n_bytes-=cuda_copy::locked_buffer_size, src+=cuda_copy::locked_buffer_size,dst+=cuda_copy::locked_buffer_size){
            auto buf = cuda_copy::locked_pool().pop();
            cuda_error_check(cudaMemcpyAsync(buf.get().data(), src, cuda_copy::locked_buffer_size, cudaMemcpyKind::cudaMemcpyDeviceToHost, cuda_stream{}));   //sync copy src device to locked
            if (async_future){async_future.wait();}
            async_future = cuda_copy::copy_pool().push_async(cuda_copy::dma_to_device, dst, buf, cuda_copy::locked_buffer_size);   //async dma transfer locked to device
        }
        if (n_bytes){
            auto buf = cuda_copy::locked_pool().pop();
            cuda_error_check(cudaMemcpyAsync(buf.get().data(), src, n_bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost, cuda_stream{}));   //sync copy src device to locked
            if (async_future){async_future.wait();}
            async_future = cuda_copy::copy_pool().push_async(cuda_copy::dma_to_device, dst, buf, n_bytes);   //async dma transfer locked to device
        }
        if (async_future){async_future.wait();}
        return d_first+n;
    }else{  //use native implementation if copy to same device or less than threshold
        return copier<native_copier_tag>::copy(first,last,d_first);
    }
}

};  //end of struct copier<multithread_copier_tag>{

}   //end of namespace cuda_copy

}   //end of namespace culib

#endif