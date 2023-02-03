#include <numeric>
#include <mmintrin.h>
#include "catch.hpp"
#include "cuda_copy.hpp"
#include "benchmark_helpers.hpp"


namespace benchmark_cuda_copy_experiment{

using cuda_experimental::device_pointer;
using cuda_experimental::locked_pointer;
using cuda_experimental::is_basic_pointer_v;
using cuda_experimental::cuda_stream;
using cuda_experimental::cuda_assert;
using cuda_experimental::cuda_copy::memcpy_avx;
using cuda_experimental::cuda_copy::locked_pool;
using cuda_experimental::cuda_copy::locked_buffer_size;
using cuda_experimental::cuda_copy::locked_uninitialized_memory;

inline auto& memcpy_workers_pool_v1(){
    static thread_pool::thread_pool_v1<void*(void*,const void*,std::size_t)> memcpy_pool{10, 10};
    return memcpy_pool;
}
inline auto& memcpy_workers_pool_v3(){
    static thread_pool::thread_pool_v3 memcpy_pool{10, 10};
    return memcpy_pool;
}

struct memcpy_pool_v1{
    auto& operator()(){return memcpy_workers_pool_v1();}
};
struct memcpy_pool_v3{
    auto& operator()(){return memcpy_workers_pool_v3();}
};

template<std::size_t, typename Pool>
auto memcpy_multithread(void*, const void*, std::size_t, void*(*)(void*,const void*,std::size_t));
template<>
auto memcpy_multithread<1, memcpy_pool_v1>(void* dst, const void* src, std::size_t n, void*(*impl)(void*,const void*,std::size_t)){impl(dst,src,n);}
template<>
auto memcpy_multithread<1, memcpy_pool_v3>(void* dst, const void* src, std::size_t n, void*(*impl)(void*,const void*,std::size_t)){impl(dst,src,n);}
template<std::size_t N, typename Pool>
auto memcpy_multithread(void* dst, const void* src, std::size_t n, void*(*impl)(void*,const void*,std::size_t)){
    static_assert(N>1);
    if (n!=0){
        auto n_chunk = n/N;
        auto n_last_chunk = n_chunk + n%N;
        auto dst_ = reinterpret_cast<unsigned char*>(dst);
        auto src_ = reinterpret_cast<const unsigned char*>(src);
        using future_type = decltype(Pool{}().push(impl,dst_,src_,n_chunk));
        std::array<future_type, N-1> futures{};
        if (n_chunk){
            for (std::size_t i{0}; i!=N-1; ++i,dst_+=n_chunk,src_+=n_chunk){
                futures[i] = Pool{}().push(impl, dst_,src_,n_chunk);
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
void copy_baseline(T* first, T* last, device_pointer<std::remove_const_t<T>> d_first){
    auto n = std::distance(first,last)*sizeof(T);
    cuda_error_check(cudaMemcpyAsync(d_first, first, n, cudaMemcpyKind::cudaMemcpyHostToDevice, cuda_stream{}));
}
template<typename T>
void copy_baseline(device_pointer<T> first, device_pointer<T> last, std::remove_const_t<T>* d_first){
    auto n = distance(first,last)*sizeof(T);
    cuda_error_check(cudaMemcpyAsync(d_first, first, n, cudaMemcpyKind::cudaMemcpyDeviceToHost, cuda_stream{}));
}

template<typename It, typename T, std::enable_if_t<!is_basic_pointer_v<It>, int> =0>
void copy_baseline(It first, It last, device_pointer<T> d_first){
    using value_type = typename std::iterator_traits<It>::value_type;
    static_assert(std::is_same_v<std::decay_t<T>,value_type>);
    auto n = std::distance(first, last);
    locked_uninitialized_memory buf{n*sizeof(value_type)};
    //check alignment
    std::uninitialized_copy_n(first, n, reinterpret_cast<value_type*>(buf.data().get()));
    cuda_error_check(cudaMemcpyAsync(d_first, buf.data(), n*sizeof(value_type), cudaMemcpyKind::cudaMemcpyHostToDevice, cuda_stream{}));
}
template<typename T, typename It, std::enable_if_t<!is_basic_pointer_v<It>,int> =0>
void copy_baseline(device_pointer<T> first, device_pointer<T> last, It d_first){
    using value_type = typename std::iterator_traits<It>::value_type;
    static_assert(std::is_same_v<std::decay_t<T>,value_type>);
    auto n = std::distance(first, last);
    locked_uninitialized_memory buf{n*sizeof(value_type)};
    cuda_error_check(cudaMemcpyAsync(buf.data(), first, n*sizeof(value_type), cudaMemcpyKind::cudaMemcpyDeviceToHost, cuda_stream{}));
    std::uninitialized_copy_n(reinterpret_cast<value_type*>(buf.data().get()), n, d_first);
}

template<typename It, typename T, std::enable_if_t<!is_basic_pointer_v<It>, int> =0>
void copy_baseline_v1(It first, It last, device_pointer<T> d_first){
    using value_type = typename std::iterator_traits<It>::value_type;
    static_assert(std::is_same_v<std::decay_t<T>,value_type>);
    auto n = std::distance(first, last);
    auto n_buffer = locked_buffer_size/sizeof(value_type);
    for(;n >= n_buffer; n-=n_buffer, first+=n_buffer, d_first+=n_buffer ){
        auto buf = locked_pool().pop();
        //check alignment
        std::uninitialized_copy_n(first, n_buffer, reinterpret_cast<value_type*>(buf.get().data().get()));
        cuda_error_check(cudaMemcpyAsync(d_first, buf.get().data(), n_buffer*sizeof(value_type), cudaMemcpyKind::cudaMemcpyHostToDevice, cuda_stream{}));
    }
    if(n){
        auto buf = locked_pool().pop();
        //check alignment
        std::uninitialized_copy_n(first, n, reinterpret_cast<value_type*>(buf.get().data().get()));
        cuda_error_check(cudaMemcpyAsync(d_first, buf.get().data(), n*sizeof(value_type), cudaMemcpyKind::cudaMemcpyHostToDevice, cuda_stream{}));
    }
}

template<typename T, typename It, std::enable_if_t<!is_basic_pointer_v<It>,int> =0>
void copy_baseline_v1(device_pointer<T> first, device_pointer<T> last, It d_first){
    using value_type = typename std::iterator_traits<It>::value_type;
    static_assert(std::is_same_v<std::decay_t<T>,value_type>);
    auto n = std::distance(first, last);

    auto n_buffer = locked_buffer_size/sizeof(value_type);

    for(;n >= n_buffer; n-=n_buffer, first+=n_buffer, d_first+=n_buffer ){
        auto buf = locked_pool().pop();
        cuda_error_check(cudaMemcpyAsync(buf.get().data(), first, n_buffer*sizeof(value_type), cudaMemcpyKind::cudaMemcpyDeviceToHost, cuda_stream{}));
        std::uninitialized_copy_n(reinterpret_cast<value_type*>(buf.get().data().get()), n_buffer, d_first);
    }
    if(n){
        auto buf = locked_pool().pop();
        cuda_error_check(cudaMemcpyAsync(buf.get().data(), first, n*sizeof(value_type), cudaMemcpyKind::cudaMemcpyDeviceToHost, cuda_stream{}));
        std::uninitialized_copy_n(reinterpret_cast<value_type*>(buf.get().data().get()), n, d_first);
    }
}


//multithread copy
inline constexpr std::size_t copy_workers = 4;
//pageable locked
template<typename Pool = memcpy_pool_v1, typename T>
void copy_multithread(const T* first, const T* last, T* d_first){
    auto n = std::distance(first,last)*sizeof(T);
    memcpy_multithread<copy_workers, Pool>(d_first,first,n,std::memcpy);
}
template<typename Pool = memcpy_pool_v1, typename T>
void copy_avx_multithread(const T* first, const T* last, T* d_first){
    auto n = std::distance(first,last)*sizeof(T);
    memcpy_multithread<copy_workers, Pool>(d_first,first,n,memcpy_avx);
}
template<typename Pool = memcpy_pool_v1, typename T>
void copy_multithread(const T* first, const T* last, locked_pointer<std::remove_const_t<T>> d_first){
    auto n = std::distance(first,last)*sizeof(T);
    memcpy_multithread<copy_workers, Pool>(d_first,first,n,host_memcpy);
}
template<typename Pool = memcpy_pool_v1, typename T>
void copy_multithread(locked_pointer<T> first, locked_pointer<T> last, std::remove_const_t<T>* d_first){
    auto n = std::distance(first,last)*sizeof(T);
    memcpy_multithread<copy_workers, Pool>(d_first,first,n,host_memcpy);
}
//locked device
template<typename Pool = memcpy_pool_v1, typename T>
void copy_multithread(locked_pointer<T> first, locked_pointer<T> last, device_pointer<std::remove_const_t<T>> d_first){
    auto n = std::distance(first,last)*sizeof(T);
    memcpy_multithread<copy_workers, Pool>(d_first,first,n,host_to_device_memcpy);
}
template<typename Pool = memcpy_pool_v1, typename T>
void copy_multithread(device_pointer<T> first, device_pointer<T> last, locked_pointer<std::remove_const_t<T>> d_first){
    auto n = distance(first,last)*sizeof(T);
    memcpy_multithread<copy_workers, Pool>(d_first,first,n,device_to_host_memcpy);
}
//host device
template<typename Pool = memcpy_pool_v1, typename T>
void copy_multithread(const T* first, const T* last, device_pointer<std::remove_const_t<T>> d_first){
    auto n = std::distance(first,last)*sizeof(T);
    memcpy_multithread<copy_workers, Pool>(d_first,first,n,host_to_device_memcpy);
}
template<typename Pool = memcpy_pool_v1, typename T>
void copy_multithread(device_pointer<T> first, device_pointer<T> last, std::remove_const_t<T>* d_first){
    auto n = distance(first,last)*sizeof(T);
    memcpy_multithread<copy_workers, Pool>(d_first,first,n,device_to_host_memcpy);
}

}   //end of namespace benchmark_cuda_copy_experiment


// TEST_CASE("benchmark_memcpy_avx","[benchmark_memcpy_avx]"){
//     using cuda_experimental::cuda_copy::memcpy_avx;
//     using benchmark_cuda_copy_experiment::copy_avx_multithread;
//     using benchmark_cuda_copy_experiment::copy_multithread;
//     using benchmark_cuda_copy_experiment::memcpy_pool_v1;
//     using benchmark_cuda_copy_experiment::memcpy_pool_v3;
//     using benchmark_helpers::make_sizes;
//     using value_type = int;
//     using host_allocator_type = std::allocator<value_type>;
//     using cuda_experimental::cpu_timer;
//     using benchmark_helpers::bandwidth_to_str;
//     using benchmark_helpers::size_to_str;

//     host_allocator_type host_alloc{};
//     //constexpr std::size_t initial_size{1000*1000+1};
//     constexpr std::size_t initial_size{1000+1};
//     constexpr std::size_t factor{2};
//     constexpr std::size_t n{15};
//     constexpr auto sizes = make_sizes<initial_size,factor,n>();
//     constexpr std::size_t iters_per_size{30};

//     for (const auto& size : sizes){
//         float dt_baseline_memcpy_ms{0};
//         float dt_avx_memcpy_ms{0};
//         auto n = size*sizeof(value_type);
//         for (std::size_t i{0}; i!=iters_per_size; ++i){
//             auto host_src_ptr = host_alloc.allocate(size);
//             auto host_dst_ptr = host_alloc.allocate(size);
//             std::iota(host_src_ptr, host_src_ptr+size, value_type{0});
//             cpu_timer start_baseline_memcpy{};
//             //std::memcpy(host_dst_ptr, host_src_ptr, n);
//             copy_multithread<memcpy_pool_v1>(host_src_ptr, host_src_ptr+size, host_dst_ptr);
//             cpu_timer stop_baseline_memcpy{};
//             dt_baseline_memcpy_ms += stop_baseline_memcpy - start_baseline_memcpy;
//             REQUIRE(std::equal(host_src_ptr, host_src_ptr+size, host_dst_ptr));
//             host_alloc.deallocate(host_dst_ptr,size);
//             host_alloc.deallocate(host_src_ptr,size);

//             host_src_ptr = host_alloc.allocate(size);
//             host_dst_ptr = host_alloc.allocate(size);
//             std::iota(host_src_ptr, host_src_ptr+size, value_type{0});
//             cpu_timer start_avx_memcpy{};
//             //memcpy_avx(host_dst_ptr, host_src_ptr, n);
//             //copy_avx_multithread(host_src_ptr, host_src_ptr+size, host_dst_ptr);
//             copy_multithread<memcpy_pool_v3>(host_src_ptr, host_src_ptr+size, host_dst_ptr);
//             cpu_timer stop_avx_memcpy{};
//             dt_avx_memcpy_ms += stop_avx_memcpy - start_avx_memcpy;
//             REQUIRE(std::equal(host_src_ptr, host_src_ptr+size, host_dst_ptr));
//             host_alloc.deallocate(host_dst_ptr,size);
//             host_alloc.deallocate(host_src_ptr,size);
//         }
//         std::cout<<std::endl<<size_to_str<value_type>(size)<<" baseline_memcpy "<<bandwidth_to_str<value_type>(size*iters_per_size, dt_baseline_memcpy_ms)
//             <<" avx_memcpy "<<bandwidth_to_str<value_type>(size*iters_per_size, dt_avx_memcpy_ms);
//     }
// }


// TEST_CASE("benchmark_baseline_copy_iterator","[benchmark_cuda_copy_experiment]"){
//     using benchmark_helpers::make_sizes;
//     using benchmark_helpers::size_to_str;
//     using benchmark_helpers::bandwidth_to_str;
//     using benchmark_cuda_copy_experiment::copy_baseline;
//     using benchmark_cuda_copy_experiment::copy_baseline_v1;
//     using benchmark_cuda_copy_experiment::copy_multithread;
//     using cuda_experimental::cuda_timer;
//     using value_type = int;
//     using device_allocator_type = cuda_experimental::device_allocator<value_type>;
//     using pageable_allocator_type = std::allocator<value_type>;
//     using locked_allocator_type = cuda_experimental::locked_allocator<value_type>;

//     constexpr std::size_t initial_size{1<<20};
//     constexpr std::size_t factor{2};
//     constexpr std::size_t n{12};
//     constexpr auto sizes = make_sizes<initial_size,factor,n>();
//     constexpr std::size_t iters_per_size{10};
//     using container_type = std::vector<value_type>;

//     device_allocator_type device_alloc{};
//     //using host_allocator_type = locked_allocator_type;
//     using host_allocator_type = pageable_allocator_type;
//     host_allocator_type host_alloc{};

//     for (const auto& size : sizes){
//         float dt_ms_to_device{0};
//         float dt_ms_to_host{0};
//         for (std::size_t i{0}; i!=iters_per_size; ++i){
//             auto device_ptr = device_alloc.allocate(size);
//             container_type host_src(size);
//             std::iota(host_src.begin(), host_src.end(), value_type{0});
//             cuda_timer start_to_device{};
//             //copy(host_src.begin(),host_src.end(),device_ptr);
//             copy_baseline_v1(host_src.begin(),host_src.end(),device_ptr);
//             //copy_baseline(host_src.begin(),host_src.end(),device_ptr);
//             cuda_timer stop_to_device{};
//             dt_ms_to_device += stop_to_device - start_to_device;
//             container_type host_dst(size);
//             std::fill(host_dst.begin(), host_dst.end(),0);
//             cuda_timer start_to_host{};
//             //copy(device_ptr, device_ptr+size, host_dst.begin());
//             copy_baseline_v1(device_ptr, device_ptr+size, host_dst.begin());
//             //copy_baseline(device_ptr, device_ptr+size, host_dst.begin());
//             cuda_timer stop_to_host{};
//             dt_ms_to_host += stop_to_host - start_to_host;
//             REQUIRE(std::equal(host_src.begin(), host_src.end(), host_dst.begin()));
//             device_alloc.deallocate(device_ptr,size);
//         }
//         std::cout<<std::endl<<size_to_str<value_type>(size)<<" to_device "<<bandwidth_to_str<value_type>(size*iters_per_size, dt_ms_to_device)<<
//             " to_host "<<bandwidth_to_str<value_type>(size*iters_per_size, dt_ms_to_host);
//         //std::cout<<std::endl<<size_to_str<value_type>(size)<<" "<<bandwidth_to_str<value_type>(size*iters_per_size, dt_ms_to_device)<<" copy timer "<<dt_ms_to_device/iters_per_size;
//     }
// }

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

TEMPLATE_TEST_CASE("benchmark_copy_to_iterator","[benchmark_cuda_copy_experiment]",
    std::vector<int>,
    std::list<int>
)
{

    using benchmark_helpers::make_sizes;
    using benchmark_helpers::size_to_str;
    using benchmark_helpers::bandwidth_to_str;
    using cuda_experimental::cpu_timer;
    using container_type = TestType;
    using value_type = container_type::value_type;
    using host_alloc_type = std::allocator<value_type>;

    constexpr std::size_t initial_size{1<<10};
    constexpr std::size_t factor{2};
    constexpr std::size_t n{20};
    constexpr auto sizes = make_sizes<initial_size,factor,n>();
    constexpr std::size_t iters_per_size{10};
    host_alloc_type host_alloc{};

    for (const auto& size : sizes){
        float dt_ms_copy{0};
        for (std::size_t i{0}; i!=iters_per_size; ++i){
            auto src = host_alloc.allocate(size);
            std::iota(src, src+size, value_type{0});
            container_type dst(size);
            cpu_timer start_copy{};
            std::copy_n(src,size,dst.begin());
            cpu_timer stop_copy{};
            dt_ms_copy += stop_copy - start_copy;
            host_alloc.deallocate(src,size);
        }
        std::cout<<std::endl<<size_to_str<value_type>(size)<<" copy "<<bandwidth_to_str<value_type>(size*iters_per_size, dt_ms_copy);
    }
}


