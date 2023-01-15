#include <numeric>
#include "catch.hpp"
#include "cuda_memcpy.hpp"
//#include "cuda_copy.hpp"
#include "benchmark_helpers.hpp"


namespace benchmark_cuda_copy_experiment{

using cuda_experimental::device_pointer;
using cuda_experimental::cuda_stream;
using cuda_experimental::cuda_assert;


inline auto& memcpy_workers_pool(){
    static thread_pool::thread_pool_v1<void(void*,void*,std::size_t)> memcpy_pool{10, 10};
    return memcpy_pool;
}

template<std::size_t>
auto memcpy_multithread(void*, void*, std::size_t, void(*)(void*,void*,std::size_t));
template<>
auto memcpy_multithread<1>(void* dst, void* src, std::size_t n, void(*impl)(void*,void*,std::size_t)){
    impl(dst,src,n);
}
template<std::size_t N>
auto memcpy_multithread(void* dst, void* src, std::size_t n, void(*impl)(void*,void*,std::size_t)){
    static_assert(N>1);
    if (n!=0){
        std::array<std::remove_reference_t<decltype(memcpy_workers_pool())>::future_type, N-1> futures{};
        auto n_chunk = n/N;
        auto n_last_chunk = n_chunk + n%N;
        auto dst_ = reinterpret_cast<unsigned char*>(dst);
        auto src_ = reinterpret_cast<unsigned char*>(src);
        if (n_chunk){
            for (std::size_t i{0}; i!=N-1; ++i,dst_+=n_chunk,src_+=n_chunk){
                futures[i] = memcpy_workers_pool().push(impl, dst_,src_,n_chunk);
            }
        }
        impl(dst_,src_,n_last_chunk);
    }
}

inline auto host_memcpy(void* dst_host, void* src_host, std::size_t n){
    std::memcpy(dst_host,src_host,n);

    //cudaMemcpy gives worse results in multithread copy than memcpy, in single thread results the same ???
    //cuda_error_check(cudaMemcpy(dst_host, src_host, n, cudaMemcpyKind::cudaMemcpyHostToHost));
}

//host to host
template<typename T>
void copy_baseline(const T* first, const T* last, T* d_first){
    auto n = std::distance(first,last)*sizeof(T);
    //std::memcpy(d_first, first, n);
    //cuda_error_check(cudaMemcpy(d_first, first, n, cudaMemcpyKind::cudaMemcpyHostToHost));
    memcpy_multithread<4>(d_first,const_cast<T*>(first),n,host_memcpy);
}
//host to device
template<typename T>
void copy_baseline(const T* first, const T* last, device_pointer<T> d_first){
    auto n = std::distance(first,last)*sizeof(T);
    cuda_error_check(cudaMemcpyAsync(static_cast<void*>(d_first.get()), static_cast<const void*>(first), n, cudaMemcpyKind::cudaMemcpyHostToDevice, cuda_stream{}));
}
//device to host
template<typename T>
void copy_baseline(device_pointer<T> first, device_pointer<T> last, std::remove_const_t<T>* d_first){
    auto n = distance(first,last)*sizeof(T);
    cuda_error_check(cudaMemcpyAsync(static_cast<void*>(d_first), static_cast<const void*>(first.get()), n, cudaMemcpyKind::cudaMemcpyDeviceToHost, cuda_stream{}));
}

}   //end of namespace benchmark_cuda_copy_experiment


// TEST_CASE("benchmark_locked_device_copy","[benchmark_cuda_copy_experiment]"){
//     using benchmark_helpers::make_sizes;
//     using benchmark_helpers::size_to_str;
//     using benchmark_helpers::bandwidth_to_str;
//     using cuda_experimental::cuda_timer;
//     using value_type = int;
//     using device_alloc_type = cuda_experimental::device_allocator<value_type>;
//     using host_alloc_type = std::allocator<value_type>;
//     using locked_alloc_type = cuda_experimental::locked_allocator<value_type>;
//     using benchmark_cuda_copy_experiment::copy_baseline;

//     constexpr std::size_t initial_size{1<<20};
//     constexpr std::size_t factor{2};
//     constexpr std::size_t n{10};
//     constexpr auto sizes = make_sizes<initial_size,factor,n>();
//     constexpr std::size_t iters_per_size{10};

//     device_alloc_type device_alloc{};
//     host_alloc_type host_alloc{};
//     locked_alloc_type locked_alloc{};

//     for (const auto& size : sizes){
//         float dt_ms_locked_to_device{0};
//         float dt_ms_device_to_locked{0};
//         for (std::size_t i{0}; i!=iters_per_size; ++i){
//             auto device_ptr = device_alloc.allocate(size);
//             auto locked_src_ptr = locked_alloc.allocate(size);
//             std::iota(locked_src_ptr.get(), (locked_src_ptr+size).get(), value_type{0});
//             cuda_timer start_locked_to_device{};
//             copy_baseline(locked_src_ptr.get(),(locked_src_ptr+size).get(),device_ptr);
//             cuda_timer stop_locked_to_device{};
//             dt_ms_locked_to_device += stop_locked_to_device - start_locked_to_device;

//             auto locked_dst_ptr = locked_alloc.allocate(size);
//             std::fill(locked_dst_ptr.get(),(locked_dst_ptr+size).get(),0);
//             cuda_timer start_device_to_locked{};
//             copy_baseline(device_ptr, device_ptr+size, locked_dst_ptr.get());
//             cuda_timer stop_device_to_locked{};
//             dt_ms_device_to_locked += stop_device_to_locked - start_device_to_locked;
//             REQUIRE(std::equal(locked_src_ptr.get(), (locked_src_ptr+size).get(), locked_dst_ptr.get()));
//             locked_alloc.deallocate(locked_dst_ptr,size);
//             locked_alloc.deallocate(locked_src_ptr,size);
//             device_alloc.deallocate(device_ptr,size);
//         }
//         std::cout<<std::endl<<size_to_str<value_type>(size)<<" locked_to_device "<<bandwidth_to_str<value_type>(size*iters_per_size, dt_ms_locked_to_device)<<
//             " device_to_locked "<<bandwidth_to_str<value_type>(size*iters_per_size, dt_ms_device_to_locked);
//     }
// }

TEST_CASE("benchmark_locked_pageable_copy","[benchmark_cuda_copy_experiment]"){

    using benchmark_helpers::make_sizes;
    using benchmark_helpers::size_to_str;
    using benchmark_helpers::bandwidth_to_str;
    using cuda_experimental::cpu_timer;
    using value_type = int;
    using device_alloc_type = cuda_experimental::device_allocator<value_type>;
    using host_alloc_type = std::allocator<value_type>;
    using locked_alloc_type = cuda_experimental::locked_allocator<value_type>;
    using benchmark_cuda_copy_experiment::copy_baseline;

    constexpr std::size_t initial_size{1<<20};
    constexpr std::size_t factor{2};
    constexpr std::size_t n{10};
    constexpr auto sizes = make_sizes<initial_size,factor,n>();
    constexpr std::size_t iters_per_size{1};

    device_alloc_type device_alloc{};
    host_alloc_type host_alloc{};
    locked_alloc_type locked_alloc{};
    //locked_alloc_type locked_alloc{cudaHostAllocWriteCombined};

    for (const auto& size : sizes){
        float dt_ms_locked_to_pageable{0};
        float dt_ms_pageable_to_locked{0};
        auto n = size*sizeof(value_type);
        for (std::size_t i{0}; i!=iters_per_size; ++i){
            auto pageable_ptr = host_alloc.allocate(size);
            auto locked_src_ptr = locked_alloc.allocate(size);
            std::iota(locked_src_ptr.get(), (locked_src_ptr+size).get(), value_type{0});
            cpu_timer start_locked_to_pageable{};
            copy_baseline(locked_src_ptr.get(),(locked_src_ptr+size).get(),pageable_ptr);
            cpu_timer stop_locked_to_pageable{};
            dt_ms_locked_to_pageable += stop_locked_to_pageable - start_locked_to_pageable;

            auto locked_dst_ptr = locked_alloc.allocate(size);
            std::fill(locked_dst_ptr.get(),(locked_dst_ptr+size).get(),0);
            cpu_timer start_pageable_to_locked{};
            copy_baseline(pageable_ptr, pageable_ptr+size, locked_dst_ptr.get());
            cpu_timer stop_pageable_to_locked{};
            dt_ms_pageable_to_locked += stop_pageable_to_locked - start_pageable_to_locked;
            REQUIRE(std::equal(locked_src_ptr.get(), (locked_src_ptr+size).get(), locked_dst_ptr.get()));
            locked_alloc.deallocate(locked_dst_ptr,size);
            locked_alloc.deallocate(locked_src_ptr,size);
            host_alloc.deallocate(pageable_ptr,size);
        }
        std::cout<<std::endl<<size_to_str<value_type>(size)<<" locked_to_pageable "<<bandwidth_to_str<value_type>(size*iters_per_size, dt_ms_locked_to_pageable)<<
            " pageable_to_locked "<<bandwidth_to_str<value_type>(size*iters_per_size, dt_ms_pageable_to_locked);
    }
}