#include <numeric>
#include <mmintrin.h>
#include "catch.hpp"
//#include "cuda_memcpy.hpp"
#include "cuda_copy.hpp"
#include "benchmark_helpers.hpp"


namespace benchmark_cuda_copy_experiment{

using cuda_experimental::device_pointer;
using cuda_experimental::locked_pointer;
using cuda_experimental::cuda_stream;
using cuda_experimental::cuda_assert;
using cuda_experimental::cuda_memcpy::memcpy_avx;


inline auto& memcpy_workers_pool(){
    static thread_pool::thread_pool_v1<void*(void*,const void*,std::size_t)> memcpy_pool{10, 10};
    return memcpy_pool;
}

template<std::size_t>
auto memcpy_multithread(void*, const void*, std::size_t, void*(*)(void*,const void*,std::size_t));
template<>
auto memcpy_multithread<1>(void* dst, const void* src, std::size_t n, void*(*impl)(void*,const void*,std::size_t)){
    impl(dst,src,n);
}
template<std::size_t N>
auto memcpy_multithread(void* dst, const void* src, std::size_t n, void*(*impl)(void*,const void*,std::size_t)){
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
}

inline auto host_memcpy(void* dst_host, const void* src_host, std::size_t n){
    //return std::memcpy(dst_host,src_host,n);
    return memcpy_avx(dst_host,src_host,n);
}
inline auto host_to_device_memcpy(void* dst_device, const void* src_host, std::size_t n){
    cuda_error_check(cudaMemcpyAsync(dst_device, src_host, n, cudaMemcpyKind::cudaMemcpyHostToDevice, cuda_stream{}));
}
inline auto device_to_host_memcpy(void* dst_host, const void* src_device, std::size_t n){
    cuda_error_check(cudaMemcpyAsync(dst_host, src_device, n, cudaMemcpyKind::cudaMemcpyDeviceToHost, cuda_stream{}));
}

//baseline copy
//host to host
template<typename T>
void copy_baseline(const T* first, const T* last, T* d_first){
    auto n = std::distance(first,last)*sizeof(T);
    std::memcpy(d_first, first, n);
}
//locked device
template<typename T>
void copy_baseline(locked_pointer<T> first, locked_pointer<T> last, device_pointer<std::remove_const_t<T>> d_first){
    auto n = std::distance(first,last)*sizeof(T);
    cuda_error_check(cudaMemcpyAsync(d_first, first, n, cudaMemcpyKind::cudaMemcpyHostToDevice, cuda_stream{}));
}
template<typename T>
void copy_baseline(device_pointer<T> first, device_pointer<T> last, locked_pointer<std::remove_const_t<T>> d_first){
    auto n = distance(first,last)*sizeof(T);
    cuda_error_check(cudaMemcpyAsync(d_first, first, n, cudaMemcpyKind::cudaMemcpyDeviceToHost, cuda_stream{}));
}
//host device
template<typename T>
void copy_baseline(const T* first, const T* last, device_pointer<std::remove_const_t<T>> d_first){
    auto n = std::distance(first,last)*sizeof(T);
    cuda_error_check(cudaMemcpyAsync(d_first, first, n, cudaMemcpyKind::cudaMemcpyHostToDevice, cuda_stream{}));
}
template<typename T>
void copy_baseline(device_pointer<T> first, device_pointer<T> last, std::remove_const_t<T>* d_first){
    auto n = distance(first,last)*sizeof(T);
    cuda_error_check(cudaMemcpyAsync(d_first, first, n, cudaMemcpyKind::cudaMemcpyDeviceToHost, cuda_stream{}));
}

//multithread copy
inline constexpr std::size_t copy_workers = 4;
//pageable locked
template<typename T>
void copy_multithread(const T* first, const T* last, T* d_first){
    auto n = std::distance(first,last)*sizeof(T);
    memcpy_multithread<copy_workers>(d_first,first,n,std::memcpy);
}
template<typename T>
void copy_avx_multithread(const T* first, const T* last, T* d_first){
    auto n = std::distance(first,last)*sizeof(T);
    memcpy_multithread<copy_workers>(d_first,first,n,memcpy_avx);
}
template<typename T>
void copy_multithread(const T* first, const T* last, locked_pointer<std::remove_const_t<T>> d_first){
    auto n = std::distance(first,last)*sizeof(T);
    memcpy_multithread<copy_workers>(d_first,first,n,host_memcpy);
}
template<typename T>
void copy_multithread(locked_pointer<T> first, locked_pointer<T> last, std::remove_const_t<T>* d_first){
    auto n = std::distance(first,last)*sizeof(T);
    memcpy_multithread<copy_workers>(d_first,first,n,host_memcpy);
}
//locked device
template<typename T>
void copy_multithread(locked_pointer<T> first, locked_pointer<T> last, device_pointer<std::remove_const_t<T>> d_first){
    auto n = std::distance(first,last)*sizeof(T);
    memcpy_multithread<copy_workers>(d_first,first,n,host_to_device_memcpy);
}
template<typename T>
void copy_multithread(device_pointer<T> first, device_pointer<T> last, locked_pointer<std::remove_const_t<T>> d_first){
    auto n = distance(first,last)*sizeof(T);
    memcpy_multithread<copy_workers>(d_first,first,n,device_to_host_memcpy);
}
//host device
template<typename T>
void copy_multithread(const T* first, const T* last, device_pointer<std::remove_const_t<T>> d_first){
    auto n = std::distance(first,last)*sizeof(T);
    memcpy_multithread<copy_workers>(d_first,first,n,host_to_device_memcpy);
}
template<typename T>
void copy_multithread(device_pointer<T> first, device_pointer<T> last, std::remove_const_t<T>* d_first){
    auto n = distance(first,last)*sizeof(T);
    memcpy_multithread<copy_workers>(d_first,first,n,device_to_host_memcpy);
}

}   //end of namespace benchmark_cuda_copy_experiment


TEST_CASE("benchmark_memcpy_avx","[benchmark_memcpy_avx]"){
    using cuda_experimental::cuda_memcpy::memcpy_avx;
    using benchmark_cuda_copy_experiment::copy_avx_multithread;
    using benchmark_cuda_copy_experiment::copy_multithread;
    using benchmark_helpers::make_sizes;
    using value_type = int;
    using host_allocator_type = std::allocator<value_type>;
    using cuda_experimental::cpu_timer;
    using benchmark_helpers::bandwidth_to_str;
    using benchmark_helpers::size_to_str;

    host_allocator_type host_alloc{};
    constexpr std::size_t initial_size{1000*1000+1};
    constexpr std::size_t factor{2};
    constexpr std::size_t n{10};
    constexpr auto sizes = make_sizes<initial_size,factor,n>();
    constexpr std::size_t iters_per_size{10};

    for (const auto& size : sizes){
        float dt_baseline_memcpy_ms{0};
        float dt_avx_memcpy_ms{0};
        auto n = size*sizeof(value_type);
        for (std::size_t i{0}; i!=iters_per_size; ++i){
            auto host_src_ptr = host_alloc.allocate(size);
            auto host_dst_ptr = host_alloc.allocate(size);
            std::iota(host_src_ptr, host_src_ptr+size, value_type{0});
            cpu_timer start_baseline_memcpy{};
            //std::memcpy(host_dst_ptr, host_src_ptr, n);
            copy_multithread(host_src_ptr, host_src_ptr+size, host_dst_ptr);
            cpu_timer stop_baseline_memcpy{};
            dt_baseline_memcpy_ms += stop_baseline_memcpy - start_baseline_memcpy;
            REQUIRE(std::equal(host_src_ptr, host_src_ptr+size, host_dst_ptr));
            host_alloc.deallocate(host_dst_ptr,size);
            host_alloc.deallocate(host_src_ptr,size);

            host_src_ptr = host_alloc.allocate(size);
            host_dst_ptr = host_alloc.allocate(size);
            std::iota(host_src_ptr, host_src_ptr+size, value_type{0});
            cpu_timer start_avx_memcpy{};
            //memcpy_avx(host_dst_ptr, host_src_ptr, n);
            copy_avx_multithread(host_src_ptr, host_src_ptr+size, host_dst_ptr);
            cpu_timer stop_avx_memcpy{};
            dt_avx_memcpy_ms += stop_avx_memcpy - start_avx_memcpy;
            REQUIRE(std::equal(host_src_ptr, host_src_ptr+size, host_dst_ptr));
            host_alloc.deallocate(host_dst_ptr,size);
            host_alloc.deallocate(host_src_ptr,size);
        }
        std::cout<<std::endl<<size_to_str<value_type>(size)<<" baseline_memcpy "<<bandwidth_to_str<value_type>(size*iters_per_size, dt_baseline_memcpy_ms)
            <<" avx_memcpy "<<bandwidth_to_str<value_type>(size*iters_per_size, dt_avx_memcpy_ms);
    }
}


// TEST_CASE("benchmark_locked_device_copy","[benchmark_cuda_copy_experiment]"){
//     using benchmark_helpers::make_sizes;
//     using benchmark_helpers::size_to_str;
//     using benchmark_helpers::bandwidth_to_str;
//     using benchmark_cuda_copy_experiment::copy_baseline;
//     using benchmark_cuda_copy_experiment::copy_multithread;
//     using cuda_experimental::cuda_timer;
//     using value_type = int;
//     using device_allocator_type = cuda_experimental::device_allocator<value_type>;
//     using pageable_allocator_type = std::allocator<value_type>;
//     using locked_allocator_type = cuda_experimental::locked_allocator<value_type>;

//     constexpr std::size_t initial_size{1<<20};
//     constexpr std::size_t factor{2};
//     constexpr std::size_t n{10};
//     constexpr auto sizes = make_sizes<initial_size,factor,n>();
//     constexpr std::size_t iters_per_size{1};

//     device_allocator_type device_alloc{};
//     //using host_allocator_type = locked_allocator_type;
//     using host_allocator_type = pageable_allocator_type;
//     host_allocator_type host_alloc{};

//     for (const auto& size : sizes){
//         float dt_ms_host_to_device{0};
//         float dt_ms_device_to_host{0};
//         for (std::size_t i{0}; i!=iters_per_size; ++i){
//             auto device_ptr = device_alloc.allocate(size);
//             auto host_src_ptr = host_alloc.allocate(size);
//             std::iota(host_src_ptr, host_src_ptr+size, value_type{0});
//             cuda_timer start_host_to_device{};
//             copy_multithread(host_src_ptr,host_src_ptr+size,device_ptr);
//             cuda_timer stop_host_to_device{};
//             dt_ms_host_to_device += stop_host_to_device - start_host_to_device;

//             auto host_dst_ptr = host_alloc.allocate(size);
//             std::fill(host_dst_ptr,host_dst_ptr+size,0);
//             cuda_timer start_device_to_host{};
//             copy_multithread(device_ptr, device_ptr+size, host_dst_ptr);
//             cuda_timer stop_device_to_host{};
//             dt_ms_device_to_host += stop_device_to_host - start_device_to_host;
//             REQUIRE(std::equal(host_src_ptr, host_src_ptr+size, host_dst_ptr));
//             host_alloc.deallocate(host_dst_ptr,size);
//             host_alloc.deallocate(host_src_ptr,size);
//             device_alloc.deallocate(device_ptr,size);
//         }
//         std::cout<<std::endl<<size_to_str<value_type>(size)<<" host_to_device "<<bandwidth_to_str<value_type>(size*iters_per_size, dt_ms_host_to_device)<<
//             " device_to_host "<<bandwidth_to_str<value_type>(size*iters_per_size, dt_ms_device_to_host);
//     }
// }

// TEST_CASE("benchmark_locked_pageable_copy","[benchmark_cuda_copy_experiment]"){

//     using benchmark_helpers::make_sizes;
//     using benchmark_helpers::size_to_str;
//     using benchmark_helpers::bandwidth_to_str;
//     using cuda_experimental::cpu_timer;
//     using value_type = int;
//     using device_alloc_type = cuda_experimental::device_allocator<value_type>;
//     using host_alloc_type = std::allocator<value_type>;
//     using locked_alloc_type = cuda_experimental::locked_allocator<value_type>;
//     using benchmark_cuda_copy_experiment::copy_baseline;
//     using benchmark_cuda_copy_experiment::copy_multithread;

//     constexpr std::size_t initial_size{1<<20};
//     constexpr std::size_t factor{2};
//     constexpr std::size_t n{10};
//     constexpr auto sizes = make_sizes<initial_size,factor,n>();
//     constexpr std::size_t iters_per_size{10};

//     device_alloc_type device_alloc{};
//     host_alloc_type host_alloc{};
//     locked_alloc_type locked_alloc{};
//     //locked_alloc_type locked_alloc{cudaHostAllocWriteCombined};

//     for (const auto& size : sizes){
//         float dt_ms_locked_to_pageable{0};
//         float dt_ms_pageable_to_locked{0};
//         auto n = size*sizeof(value_type);
//         for (std::size_t i{0}; i!=iters_per_size; ++i){
//             auto pageable_ptr = host_alloc.allocate(size);
//             auto locked_src_ptr = locked_alloc.allocate(size);
//             std::iota(locked_src_ptr, locked_src_ptr+size, value_type{0});
//             cpu_timer start_locked_to_pageable{};
//             copy_multithread(locked_src_ptr,locked_src_ptr+size,pageable_ptr);
//             cpu_timer stop_locked_to_pageable{};
//             dt_ms_locked_to_pageable += stop_locked_to_pageable - start_locked_to_pageable;

//             auto locked_dst_ptr = locked_alloc.allocate(size);
//             std::fill(locked_dst_ptr, locked_dst_ptr+size, 0);
//             cpu_timer start_pageable_to_locked{};
//             copy_multithread(pageable_ptr, pageable_ptr+size, locked_dst_ptr);
//             cpu_timer stop_pageable_to_locked{};
//             dt_ms_pageable_to_locked += stop_pageable_to_locked - start_pageable_to_locked;
//             REQUIRE(std::equal(locked_src_ptr, locked_src_ptr+size, locked_dst_ptr));
//             locked_alloc.deallocate(locked_dst_ptr,size);
//             locked_alloc.deallocate(locked_src_ptr,size);
//             host_alloc.deallocate(pageable_ptr,size);
//         }
//         std::cout<<std::endl<<size_to_str<value_type>(size)<<" locked_to_pageable "<<bandwidth_to_str<value_type>(size*iters_per_size, dt_ms_locked_to_pageable)<<
//             " pageable_to_locked "<<bandwidth_to_str<value_type>(size*iters_per_size, dt_ms_pageable_to_locked);
//     }
// }