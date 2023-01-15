#ifndef CUDA_MEMCPY_HPP_
#define CUDA_MEMCPY_HPP_

#include "cuda_pointer.hpp"
#include "cuda_allocator.hpp"
#include "thread_pool.hpp"
#include "bounded_pool.hpp"

namespace cuda_experimental{

namespace cuda_memcpy{

inline constexpr std::size_t memcpy_pool_size = 4;
inline constexpr std::size_t dma_pool_size = 4;
inline constexpr std::size_t memcpy_workers = memcpy_pool_size;
inline constexpr std::size_t locked_buffer_size = 64*1024*1024;
inline constexpr std::size_t locked_pool_size = 4;
inline constexpr std::size_t copy_v2_pool_size = locked_pool_size;
inline constexpr std::size_t copy_v2_workers = copy_v2_pool_size;

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
    static thread_pool::thread_pool_v1<decltype(std::memcpy)> memcpy_pool{memcpy_pool_size, memcpy_pool_size};
    return memcpy_pool;
}

//host - host multithread memcpy
//sync wrt caller thread, utilizes N threads: caller thread +  N-1 workers from pool
//use to copy from host pageable to host locked memory
template<std::size_t> auto memcpy_multithread(void* dst, const void* src, std::size_t n);
template<>
auto memcpy_multithread<1>(void* dst, const void* src, std::size_t n){
    std::memcpy(dst,src,n);
}
template<std::size_t N>
auto memcpy_multithread(void* dst, const void* src, std::size_t n){
    static_assert(N>1);
    if (n!=0){
        std::array<std::remove_reference_t<decltype(memcpy_workers_pool())>::future_type, N-1> futures{};
        auto n_chunk = n/N;
        auto n_last_chunk = n_chunk + n%N;
        auto dst_ = reinterpret_cast<unsigned char*>(dst);
        auto src_ = reinterpret_cast<const unsigned char*>(src);
        if (n_chunk){
            for (std::size_t i{0}; i!=N-1; ++i,dst_+=n_chunk,src_+=n_chunk){
                futures[i] = memcpy_workers_pool().push(std::memcpy, dst_,src_,n_chunk);
            }
        }
        std::memcpy(dst_,src_,n_last_chunk);
    }
}

template<std::size_t N>
auto memcpy_multithread_async(void* dst, decltype(locked_pool().pop()) src_locked, std::size_t n){
    static_assert(N>1);
    std::array<std::remove_reference_t<decltype(memcpy_workers_pool())>::future_type, N> futures{};
    if (n!=0){
        auto n_chunk = n/N;
        auto n_last_chunk = n_chunk + n%N;
        auto dst_ = reinterpret_cast<unsigned char*>(dst);
        auto src_ = reinterpret_cast<const unsigned char*>(src_locked.get().data().get());
        if (n_chunk){
            for (std::size_t i{0}; i!=N-1; ++i,dst_+=n_chunk,src_+=n_chunk){
                futures[i] = memcpy_workers_pool().push_async(std::memcpy, dst_,src_,n_chunk);
            }
        }
        if (n_last_chunk){
            futures[N-1] = memcpy_workers_pool().push_async(std::memcpy, dst_,src_,n_last_chunk);
        }
    }
    return futures;
}

//dma transfer between locked and device
auto dma_to_device(device_pointer<unsigned char> dst_device, decltype(locked_pool().pop()) src_locked, std::size_t n){
    cuda_error_check(cudaMemcpyAsync(static_cast<void*>(dst_device.get()), static_cast<const void*>(src_locked.get().data().get()) , n, cudaMemcpyKind::cudaMemcpyHostToDevice, cuda_stream{}));
}
auto dma_to_host(device_pointer<unsigned char> src_device, decltype(locked_pool().pop()) dst_locked, std::size_t n){
    cuda_error_check(cudaMemcpyAsync(static_cast<void*>(dst_locked.get().data().get()), static_cast<const void*>(src_device.get()) , n, cudaMemcpyKind::cudaMemcpyDeviceToHost, cuda_stream{}));
}

inline auto& dma_workers_pool(){
    static thread_pool::thread_pool_v1<decltype(dma_to_device)> dma_pool{dma_pool_size, dma_pool_size};
    return dma_pool;
}

auto copy_to_device(device_pointer<unsigned char> dst_device, unsigned char* src_host, std::size_t n){
    auto buf = cuda_memcpy::locked_pool().pop();
    auto buf_ptr = static_cast<void*>(buf.get().data().get());
    cuda_memcpy::memcpy_multithread<cuda_memcpy::memcpy_workers>(buf_ptr,static_cast<const void*>(src_host), n);   //sync copy to locked buffer
    cuda_error_check(cudaMemcpyAsync(static_cast<void*>(dst_device.get()), buf_ptr, n, cudaMemcpyKind::cudaMemcpyHostToDevice, cuda_stream{}));    //sync dma to device
    //buffer returns to pool on exit
}
auto copy_to_host(device_pointer<unsigned char> src_device, unsigned char* dst_host, std::size_t n){
    auto buf = cuda_memcpy::locked_pool().pop();
    auto buf_ptr = static_cast<void*>(buf.get().data().get());
    cuda_error_check(cudaMemcpyAsync(buf_ptr, static_cast<const void*>(src_device.get()), n, cudaMemcpyKind::cudaMemcpyDeviceToHost, cuda_stream{}));    //sync dma to locked buffer
    cuda_memcpy::memcpy_multithread<cuda_memcpy::memcpy_workers>(static_cast<void*>(dst_host), buf_ptr, n);   //sync copy to host
    //buffer returns to pool on exit
}

inline auto& copy_v2_workers_pool(){
    static thread_pool::thread_pool_v1<decltype(copy_to_device)> copy_v2_pool{copy_v2_pool_size, copy_v2_pool_size};
    return copy_v2_pool;
}

}   //end of namespace cuda_memcpy

//host to device copy
template<typename T>
void copy(const T* first, const T* last, device_pointer<T> d_first){
    using cuda_memcpy::dma_workers_pool;
    auto n = std::distance(first,last)*sizeof(T);
    auto n_chunks = n/cuda_memcpy::locked_buffer_size;
    auto last_chunk_size = n%cuda_memcpy::locked_buffer_size;
    auto src = reinterpret_cast<const unsigned char*>(first);
    auto dst = static_cast<device_pointer<unsigned char>>(d_first);
    typename std::remove_reference_t<decltype(dma_workers_pool())>::future_type dma_future{};
    for (std::size_t i{0}; i!=n_chunks; ++i,src+=cuda_memcpy::locked_buffer_size,dst+=cuda_memcpy::locked_buffer_size){
        auto buf = cuda_memcpy::locked_pool().pop();
        cuda_memcpy::memcpy_multithread<cuda_memcpy::memcpy_workers>(static_cast<void*>(buf.get().data().get()),static_cast<const void*>(src), cuda_memcpy::locked_buffer_size);   //sync copy to locked buffer
        if (dma_future){dma_future.wait();}
        dma_future = cuda_memcpy::dma_workers_pool().push_async(cuda_memcpy::dma_to_device, dst, buf, cuda_memcpy::locked_buffer_size);   //async dma transfer to device
    }
    if (last_chunk_size){
        auto buf = cuda_memcpy::locked_pool().pop();
        cuda_memcpy::memcpy_multithread<cuda_memcpy::memcpy_workers>(static_cast<void*>(buf.get().data().get()),static_cast<const void*>(src),last_chunk_size);
        if (dma_future){dma_future.wait();}
        dma_future = cuda_memcpy::dma_workers_pool().push_async(cuda_memcpy::dma_to_device, dst, buf, last_chunk_size);
    }
    if (dma_future){dma_future.wait();}
}

template<typename T>
void copy_v2(const T* first, const T* last, device_pointer<T> d_first){
    auto n = std::distance(first,last)*sizeof(T);
    auto n_chunks = n/cuda_memcpy::locked_buffer_size;
    auto last_chunk_size = n%cuda_memcpy::locked_buffer_size;
    auto src = reinterpret_cast<unsigned char*>(const_cast<T*>(first));
    auto dst = static_cast<device_pointer<unsigned char>>(d_first);
    using future_type = std::remove_reference_t<decltype(cuda_memcpy::copy_v2_workers_pool())>::future_type;
    std::array<future_type, cuda_memcpy::copy_v2_workers> futures{};
    future_type last_future{};
    for (std::size_t i{0}; i!=n_chunks; ++i,src+=cuda_memcpy::locked_buffer_size,dst+=cuda_memcpy::locked_buffer_size){
        auto idx = i%cuda_memcpy::copy_v2_workers;
        if (idx==0){
            std::for_each(futures.begin(),futures.end(), [](auto& f){if(f){f.wait();}});
        }
        futures[idx] = cuda_memcpy::copy_v2_workers_pool().push_async(cuda_memcpy::copy_to_device, dst, src, cuda_memcpy::locked_buffer_size);
    }
    if (last_chunk_size){
        last_future = cuda_memcpy::copy_v2_workers_pool().push_async(cuda_memcpy::copy_to_device, dst, src, last_chunk_size);
    }
    std::for_each(futures.begin(),futures.end(), [](auto& f){if(f){f.wait();}});
    if (last_future){last_future.wait();}
}

//device to host copy
template<typename T>
void copy(device_pointer<T> first, device_pointer<T> last, std::remove_const_t<T>* d_first){
    using cuda_memcpy::dma_workers_pool;
    auto n = std::distance(first,last)*sizeof(T);
    auto n_chunks = n/cuda_memcpy::locked_buffer_size;
    auto last_chunk_size = n%cuda_memcpy::locked_buffer_size;
    auto src = static_cast<device_pointer<unsigned char>>(first);
    auto dst = reinterpret_cast<unsigned char*>(d_first);
    std::array<std::remove_reference_t<decltype(cuda_memcpy::memcpy_workers_pool())>::future_type, cuda_memcpy::memcpy_workers> futures{};
    for (std::size_t i{0}; i!=n_chunks; ++i,src+=cuda_memcpy::locked_buffer_size,dst+=cuda_memcpy::locked_buffer_size){
        auto buf = cuda_memcpy::locked_pool().pop();
        //cuda_memcpy::dma_workers_pool().push(cuda_memcpy::dma_to_host, src, buf, cuda_memcpy::locked_buffer_size);   //sync dma transfer from device to locked
        cuda_error_check(cudaMemcpyAsync(static_cast<void*>(buf.get().data().get()), static_cast<const void*>(src.get()) , cuda_memcpy::locked_buffer_size, cudaMemcpyKind::cudaMemcpyDeviceToHost, cuda_stream{}));
        std::for_each(futures.begin(),futures.end(),[](auto& f){if (f){f.wait();}});
        futures = cuda_memcpy::memcpy_multithread_async<cuda_memcpy::memcpy_workers>(static_cast<void*>(dst), buf, cuda_memcpy::locked_buffer_size);   //copy to pageable
    }
    if (last_chunk_size){
        auto buf = cuda_memcpy::locked_pool().pop();
        //cuda_memcpy::dma_workers_pool().push(cuda_memcpy::dma_to_host, src, buf, last_chunk_size);
        cuda_error_check(cudaMemcpyAsync(static_cast<void*>(buf.get().data().get()), static_cast<const void*>(src.get()) , last_chunk_size, cudaMemcpyKind::cudaMemcpyDeviceToHost, cuda_stream{}));
        futures = cuda_memcpy::memcpy_multithread_async<cuda_memcpy::memcpy_workers>(static_cast<void*>(dst), buf, last_chunk_size);
    }
    std::for_each(futures.begin(),futures.end(),[](auto& f){if (f){f.wait();}});
}


}   //end of namespace cuda_experimental

#endif