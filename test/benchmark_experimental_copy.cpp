#include <array>
#include <sstream>
#include <string>
#include <iostream>
#include <thread>
#include <immintrin.h>
#include "catch.hpp"
#include "cuda_memory.hpp"
#include "benchmark_helpers.hpp"

namespace benchmark_experimental_copy{

void avx_memcpy_lu_su(void* dst, const void* src, std::size_t n){
    using block_type = __m256i;
    auto src_it = reinterpret_cast<const block_type*>(src);
    auto src_end = src_it + n/sizeof(block_type);
    auto dst_it = reinterpret_cast<block_type*>(dst);
    for (; src_it!=src_end; ++src_it,++dst_it){
        //load unaligned, store to unaligned
        auto block = _mm256_loadu_si256(src_it);
        _mm256_storeu_si256(dst_it,block);
    }
}
void avx_memcpy_lu_sant(void* dst, const void* src, std::size_t n){
    using block_type = __m256i;
    auto src_it = reinterpret_cast<const block_type*>(src);
    auto src_end = src_it + n/sizeof(block_type);
    auto dst_it = reinterpret_cast<block_type*>(dst);
    for (; src_it!=src_end; ++src_it,++dst_it){
        //load unaligned, store to aligned nt
        auto block = _mm256_loadu_si256(src_it);
        _mm256_stream_si256(dst_it,block);
    }
}
void avx_memcpy_lant_su(void* dst, const void* src, std::size_t n){
    using block_type = __m256i;
    auto src_it = reinterpret_cast<const block_type*>(src);
    auto src_end = src_it + n/sizeof(block_type);
    auto dst_it = reinterpret_cast<block_type*>(dst);
    for (; src_it!=src_end; ++src_it,++dst_it){
        //load aligned nt, store unaligned
        auto block = _mm256_stream_load_si256(src_it);
        _mm256_storeu_si256(dst_it,block);
    }
}
//vmovntdq
void avx_memcpy_lant_sant(void* dst, const void* src, std::size_t n){
    using block_type = __m256i;
    auto src_it = reinterpret_cast<const block_type*>(src);
    auto src_end = src_it + n/sizeof(block_type);
    auto dst_it = reinterpret_cast<block_type*>(dst);
    for (; src_it!=src_end; ++src_it,++dst_it){
        //load aligned nt, store aligned nt
        auto block = _mm256_stream_load_si256(src_it);
        _mm256_stream_si256(dst_it,block);
    }
}
struct avx_memcpy_lu_su_test_helper
{
    static constexpr char* name = "avx_memcpy_lu_su";
    auto operator()(void* dst, const void* src, std::size_t n){avx_memcpy_lu_su(dst,src,n);}
};
struct avx_memcpy_lu_sant_test_helper
{
    static constexpr char* name = "avx_memcpy_lu_sant";
    auto operator()(void* dst, const void* src, std::size_t n){avx_memcpy_lu_sant(dst,src,n);}
};
struct avx_memcpy_lant_su_test_helper
{
    static constexpr char* name = "avx_memcpy_lant_su";
    auto operator()(void* dst, const void* src, std::size_t n){avx_memcpy_lant_su(dst,src,n);}
};
struct avx_memcpy_lant_sant_test_helper
{
    static constexpr char* name = "avx_memcpy_lant_sant";
    auto operator()(void* dst, const void* src, std::size_t n){avx_memcpy_lant_sant(dst,src,n);}
};

template<std::size_t> auto memcpy_multithread(void* dst, const void* src, std::size_t n);

template<>
auto memcpy_multithread<0>(void* dst, const void* src, std::size_t n){
    std::memcpy(dst,src,n);
}

template<std::size_t N = 1>
auto memcpy_multithread(void* dst, const void* src, std::size_t n){
    using cuda_experimental::thread_sync_wrapper;
    static_assert(N!=0);
    if (n!=0){
        std::array<thread_sync_wrapper, N> workers{};
        auto n_chunk = n/N;
        auto n_first_chunk = n_chunk + n%N;
        workers[0] = thread_sync_wrapper{std::memcpy, dst,src,n_first_chunk};
        if (n_chunk){
            auto dst_ = reinterpret_cast<unsigned char*>(dst);
            auto src_ = reinterpret_cast<const unsigned char*>(src);
            dst_+=n_first_chunk;
            src_+=n_first_chunk;
            for (std::size_t i{1}; i!=N; ++i,dst_+=n_chunk,src_+=n_chunk){
                workers[i] = thread_sync_wrapper{std::memcpy, dst_,src_,n_chunk};
            }
        }
    }
}


}


// TEST_CASE("benchmark_experimental_copy_host_device_baseline","[benchmark_experimental_copy]")
// {
//     using value_type = float;
//     using device_allocator_type = cuda_experimental::device_allocator<value_type>;
//     using cuda_experimental::cuda_timer;
//     using cuda_experimental::cpu_timer;
//     using cuda_experimental::cuda_stream;
//     using cuda_experimental::cuda_assert;
//     using cuda_experimental::ptr_to_void;
//     using benchmark_helpers::make_sizes;
//     using benchmark_helpers::size_to_str;
//     using benchmark_helpers::bandwidth_to_str;
//     using cuda_experimental::locked_buffer;
//     using cuda_experimental::registered_buffer;
//     using cuda_experimental::pageable_buffer;
//     using cuda_experimental::device_buffer;

//     constexpr std::size_t initial_size{1<<20};
//     constexpr std::size_t factor{2};
//     constexpr std::size_t n{10};
//     constexpr auto sizes = make_sizes<initial_size,factor,n>();
//     constexpr auto max_size = sizes.back();

//     //warming
//     // for (std::size_t i{0}; i!=10; ++i){
//     //     auto size = initial_size;
//     //     auto host_buffer = host_buffer_maker{}(size);
//     //     auto dev_buffer = dev_allocator.allocate(size);
//     //     copy(host_buffer.begin(), host_buffer.end(),dev_buffer);
//     //     dev_allocator.deallocate(dev_buffer,size);
//     // }

//     SECTION("benchmark_copy_host_to_device"){
//         std::cout<<std::endl<<"benchmark_copy_host_to_device_baseline"<<std::endl;
//         for (const auto& size : sizes){
//             auto host_buffer = pageable_buffer<value_type>(size);
//             std::fill(host_buffer.begin(),host_buffer.end(),11.0f);
//             auto dev_buffer = device_buffer<value_type>(size);
//             auto cpu_start = cpu_timer{};
//             auto start = cuda_timer{};
//             cuda_error_check(cudaMemcpyAsync(ptr_to_void(dev_buffer.data()), ptr_to_void(host_buffer.data()), size*sizeof(value_type), cudaMemcpyKind::cudaMemcpyHostToDevice, cuda_stream{}));
//             auto stop = cuda_timer{};
//             auto cpu_stop = cpu_timer{};
//             auto dt_ms = stop - start;
//             auto dt_cpu_ms = cpu_stop - cpu_start;
//             std::cout<<std::endl<<size_to_str<value_type>(size)<<" "<<bandwidth_to_str<value_type>(size, dt_ms)<<" cuda_timer "<<dt_ms<<" cpu_timer "<<dt_cpu_ms;
//         }
//     }


//     // SECTION("benchmark_copy_device_to_host"){
//     //     std::cout<<std::endl<<std::endl<<host_buffer_maker::name;
//     //     std::cout<<std::endl<<"benchmark_copy_device_to_host"<<std::endl;
//     //     //warming
//     //     for (std::size_t i{0}; i!=10; ++i){
//     //         auto size = initial_size;
//     //         auto host_buffer = host_buffer_maker{}(size);
//     //         auto dev_buffer = dev_allocator.allocate(size);
//     //         copy(dev_buffer, dev_buffer+size,host_buffer.begin());
//     //         dev_allocator.deallocate(dev_buffer,size);
//     //     }
//     //     //benching dev to host
//     //     for (const auto& size : sizes){
//     //         auto host_buffer = host_buffer_maker{}(size);
//     //         auto dev_buffer = dev_allocator.allocate(size);
//     //         auto start = cuda_timer{};
//     //         copy(dev_buffer, dev_buffer+size,host_buffer.begin());
//     //         auto stop = cuda_timer{};
//     //         auto dt_ms = stop - start;
//     //         std::cout<<std::endl<<size_to_str<value_type>(size)<<" "<<bandwidth_to_str<value_type>(size, dt_ms);
//     //         dev_allocator.deallocate(dev_buffer,size);
//     //     }
//     // }
// }

// TEMPLATE_TEST_CASE("benchmark_experimental_copy_host_device_full_size_buffer","[benchmark_experimental_copy]",
//     cuda_experimental::locked_buffer<float>,
//     cuda_experimental::registered_buffer<float>
// )
// {
//     using buffer_type = TestType;
//     using value_type = typename buffer_type::value_type;
//     using device_allocator_type = cuda_experimental::device_allocator<value_type>;
//     using cuda_experimental::cuda_timer;
//     using cuda_experimental::cpu_timer;
//     using cuda_experimental::cuda_stream;
//     using cuda_experimental::cuda_assert;
//     using cuda_experimental::ptr_to_void;
//     using benchmark_helpers::make_sizes;
//     using benchmark_helpers::size_to_str;
//     using benchmark_helpers::bandwidth_to_str;
//     using cuda_experimental::locked_buffer;
//     using cuda_experimental::registered_buffer;
//     using cuda_experimental::pageable_buffer;
//     using cuda_experimental::device_buffer;

//     constexpr std::size_t initial_size{1<<20};
//     constexpr std::size_t factor{2};
//     constexpr std::size_t n{10};
//     constexpr auto sizes = make_sizes<initial_size,factor,n>();
//     constexpr auto max_size = sizes.back();
//     std::cout<<std::endl<<std::endl<<"benchmark_copy_host_to_device_full_size_buffer"<<std::endl<<typeid(buffer_type).name()<<std::endl;
//     for (const auto& size : sizes){
//         auto host_buffer = pageable_buffer<value_type>(size);
//         std::fill(host_buffer.begin(),host_buffer.end(),11.0f);
//         auto dev_buffer = device_buffer<value_type>(size);
//         auto staging_start = cpu_timer{};
//         auto buffer = buffer_type(size);
//         auto copy_start = cpu_timer{};
//         std::memcpy(buffer.data().get(),host_buffer.data(), size*sizeof(value_type));
//         auto staging_stop = cpu_timer{};
//         auto device_trnsfer_start = cuda_timer{};
//         cuda_error_check(cudaMemcpyAsync(ptr_to_void(dev_buffer.data()), ptr_to_void(buffer.data()), size*sizeof(value_type), cudaMemcpyKind::cudaMemcpyHostToDevice, cuda_stream{}));
//         auto device_trnsfer_stop = cuda_timer{};
//         auto full_transfer_stop = cpu_timer{};
//         auto dt_device_transfer_ms = device_trnsfer_stop - device_trnsfer_start;
//         auto dt_full_transfer = full_transfer_stop - copy_start;
//         auto dt_allocate_ms = copy_start - staging_start;
//         auto dt_copy_ms = staging_stop - copy_start;
//         std::cout<<std::endl<<size_to_str<value_type>(size)<<" device_transfer_timer "<<bandwidth_to_str<value_type>(size, dt_device_transfer_ms)
//             <<" full_transfer_timer "<<bandwidth_to_str<value_type>(size, dt_full_transfer)<<" "<<dt_full_transfer
//             <<" allocate_timer "<<dt_allocate_ms<<" copy_timer "<<" copy bandwidth "<<bandwidth_to_str<value_type>(size, dt_copy_ms)
//             <<dt_copy_ms << " transfer timer "<<dt_device_transfer_ms;
//     }
// }

// TEST_CASE("benchmark_experimental_copy_host_device_reuse_buffer_avx_memcpy","[benchmark_experimental_copy]")
// {
//     using cuda_experimental::locked_buffer;
//     using cuda_experimental::registered_buffer;
//     using cuda_experimental::pageable_buffer;
//     using cuda_experimental::device_buffer;
//     using cuda_experimental::copy_buffer;

//     using value_type = float;
//     using device_allocator_type = cuda_experimental::device_allocator<value_type>;
//     using cuda_experimental::cuda_timer;
//     using cuda_experimental::cpu_timer;
//     using cuda_experimental::cuda_stream;
//     using cuda_experimental::aligned_for_copy;
//     using cuda_experimental::cuda_assert;
//     using cuda_experimental::ptr_to_void;
//     using benchmark_helpers::make_sizes;
//     using benchmark_helpers::size_to_str;
//     using benchmark_helpers::bandwidth_to_str;

//     constexpr std::size_t initial_copy_size{1024*1024};
//     constexpr std::size_t copy_size_factor{2};
//     constexpr std::size_t n{10};
//     constexpr auto copy_sizes = make_sizes<initial_copy_size,copy_size_factor,n>();
//     constexpr std::size_t max_copy_size = copy_sizes.back();

//     using buffer_type = locked_buffer<unsigned char>;
//     using block_type = __m256i;
//     constexpr std::size_t initial_buffer_size{1024*1024};
//     constexpr std::size_t buffer_size_factor{2};
//     constexpr std::size_t n_buffer_sizes{10};
//     constexpr auto buffer_sizes = make_sizes<initial_buffer_size,buffer_size_factor,n_buffer_sizes>();
//     auto buffer_size = buffer_sizes[6];

//     std::cout<<std::endl<<std::endl<<"benchmark_experimental_copy_host_device_reuse_buffer_avx_memcpy"<<std::endl;
//     std::cout<<std::endl<<std::endl<<"buffer size"<<size_to_str<buffer_type::value_type>(buffer_size)<<std::endl;
//     for (auto size : copy_sizes){
//         auto src = pageable_buffer<value_type>(size);
//         std::fill_n(src.data(),src.size(),11.0f);
//         auto device_dst = device_buffer<value_type>(size);

//         auto start = cpu_timer{};
//         auto buffer = buffer_type(buffer_size);
//         auto start_copy = cpu_timer{};
//         const auto src_aligned = aligned_for_copy<block_type>(src.data(),src.size()*sizeof(value_type));
//         const auto buffer_aligned = aligned_for_copy<block_type>(buffer.data().get(),buffer.size()*sizeof(buffer_type::value_type));

//         const auto n_buffers = src_aligned.aligned_blocks() / buffer_aligned.aligned_blocks();
//         const auto n_last_blocks = src_aligned.aligned_blocks() % buffer_aligned.aligned_blocks();
//         const auto n_buffer = buffer_aligned.aligned_blocks()*buffer_aligned.block_size();

//         if(src_aligned.aligned_blocks()){
//             //std::cout<<std::endl<<"if(src_aligned.aligned_blocks()){"<<src_aligned.aligned_blocks();
//             auto src_it = reinterpret_cast<block_type*>(src_aligned.first_aligned());
//             auto dst_it = reinterpret_cast<unsigned char*>(device_dst.data().get());
//             for (std::size_t i{0}; i!=n_buffers; ++i){
//                 auto buffer_it = reinterpret_cast<block_type*>(buffer_aligned.first_aligned());
//                 auto buffer_end = buffer_it + buffer_aligned.aligned_blocks();
//                 for (;buffer_it!=buffer_end; ++buffer_it,++src_it){
//                     auto block = _mm256_stream_load_si256(src_it);
//                     _mm256_stream_si256(buffer_it,block);
//                 }
//                 cuda_error_check(cudaMemcpyAsync(dst_it, buffer_aligned.first_aligned(), n_buffer, cudaMemcpyKind::cudaMemcpyHostToDevice, cuda_stream{}));
//                 dst_it+=n_buffer;
//             }
//             if (n_last_blocks){
//                 //std::cout<<std::endl<<"if (n_last_blocks){"<<n_last_blocks;
//                 auto buffer_it = reinterpret_cast<block_type*>(buffer_aligned.first_aligned());
//                 auto buffer_end = buffer_it + n_last_blocks;
//                 auto n_last_buffer = n_last_blocks*buffer_aligned.block_size();
//                 for (;buffer_it!=buffer_end; ++buffer_it,++src_it){
//                     auto block = _mm256_stream_load_si256(src_it);
//                     _mm256_stream_si256(buffer_it,block);
//                 }
//                 cuda_error_check(cudaMemcpyAsync(dst_it, buffer_aligned.first_aligned(), n_last_buffer, cudaMemcpyKind::cudaMemcpyHostToDevice, cuda_stream{}));
//                 dst_it+=n_last_buffer;
//             }
//             auto n_last_bytes = reinterpret_cast<unsigned char*>(src.data()+src.size()) - reinterpret_cast<unsigned char*>(src_it);
//             if (n_last_bytes){
//                 //std::cout<<std::endl<<"if (n_last_bytes){"<<n_last_bytes;
//                 std::memcpy(buffer_aligned.first(), src_it, n_last_bytes);
//                 cuda_error_check(cudaMemcpyAsync(dst_it, buffer_aligned.first(), n_last_bytes, cudaMemcpyKind::cudaMemcpyHostToDevice, cuda_stream{}));
//             }
//             if (src_aligned.first_offset()){
//                 //std::cout<<std::endl<<"if (src_aligned.first_offset()){"<<src_aligned.first_offset();
//                 std::memcpy(buffer_aligned.first(), src_aligned.first(), src_aligned.first_offset());
//                 cuda_error_check(cudaMemcpyAsync(device_dst.data().get(), buffer_aligned.first(), src_aligned.first_offset(), cudaMemcpyKind::cudaMemcpyHostToDevice, cuda_stream{}));
//             }
//         }else{
//             std::memcpy(buffer_aligned.first(), src_aligned.first(), src_aligned.n());
//             cuda_error_check(cudaMemcpyAsync(device_dst.data().get(), buffer_aligned.first(), src_aligned.n(), cudaMemcpyKind::cudaMemcpyHostToDevice, cuda_stream{}));
//         }
//         auto stop = cpu_timer{};
//         auto dt_ms = stop - start;
//         auto dt_buffer_ms = start_copy - start;
//         auto dt_copy_ms = stop - start_copy;
//         std::cout<<std::endl<<size_to_str<value_type>(size)<<" "<<bandwidth_to_str<value_type>(size, dt_ms)<<" "<<dt_ms<<" "<<bandwidth_to_str<value_type>(size,dt_copy_ms)<<" "<<dt_copy_ms
//         <<" buffer_timer "<<dt_buffer_ms;
//     }
// }

// TEST_CASE("benchmark_experimental_copy_host_pagable_device_buffer","[benchmark_experimental_copy]")
// {
//     using value_type = float;
//     using device_allocator_type = cuda_experimental::device_allocator<value_type>;
//     using cuda_experimental::cuda_timer;
//     using cuda_experimental::cpu_timer;
//     using cuda_experimental::cuda_stream;
//     using cuda_experimental::cuda_assert;
//     using cuda_experimental::ptr_to_void;
//     using benchmark_helpers::make_sizes;
//     using benchmark_helpers::size_to_str;
//     using benchmark_helpers::bandwidth_to_str;
//     using cuda_experimental::locked_buffer;
//     using cuda_experimental::pageable_buffer;
//     using cuda_experimental::device_buffer;
//     using cuda_experimental::copy_buffer;

//     constexpr std::size_t initial_copy_size{1024*1024};
//     constexpr std::size_t copy_size_factor{2};
//     constexpr std::size_t n{10};
//     constexpr auto copy_sizes = make_sizes<initial_copy_size,copy_size_factor,n>();
//     constexpr std::size_t max_copy_size = copy_sizes.back();

//     constexpr std::size_t initial_buffer_size{1024*1024};
//     constexpr std::size_t buffer_size_factor{2};
//     constexpr std::size_t n_buffer_sizes{10};
//     //constexpr auto buffer_sizes = make_sizes<initial_buffer_size,buffer_size_factor,n_buffer_sizes>();

//     for (auto size : copy_sizes){
//         auto host_src = pageable_buffer<value_type>(size);
//         std::fill(host_src.begin(),host_src.end(),11.0f);
//         auto device_dst = device_buffer<value_type>(size);

//         auto start = cpu_timer{};
//         auto buffer = locked_buffer<value_type>(size);
//         auto start_copy = cpu_timer{};
//         auto n = size*sizeof(value_type);
//         std::memcpy(buffer.data().get(), host_src.data(), n);
//         cuda_error_check(cudaMemcpyAsync(device_dst.data().get(), buffer.data().get(), n, cudaMemcpyKind::cudaMemcpyHostToDevice, cuda_stream{}));
//         auto stop = cpu_timer{};

//         auto dt_ms = stop - start;
//         auto dt_copy_ms = stop - start_copy;
//         std::cout<<std::endl<<size_to_str<value_type>(size)<<" "<<bandwidth_to_str<value_type>(size, dt_ms)<<" "<<dt_ms<<" "<<bandwidth_to_str<value_type>(size, dt_copy_ms)<<" "<<dt_copy_ms;
//     }
// }



TEST_CASE("benchmark_experimental_copy_host_host","[benchmark_experimental_copy]")
{
    using value_type = float;
    using cuda_experimental::cpu_timer;
    using cuda_experimental::ptr_to_void;
    using cuda_experimental::alignment;
    using benchmark_helpers::make_sizes;
    using benchmark_helpers::size_to_str;
    using benchmark_helpers::bandwidth_to_str;
    using cuda_experimental::locked_buffer;
    using cuda_experimental::pageable_buffer;
    using cuda_experimental::locked_allocator;
    using cuda_experimental::aligned_for_copy;


    static_assert(std::is_trivially_copyable_v<value_type>);
    constexpr std::size_t initial_size{1<<20};
    constexpr std::size_t factor{2};
    constexpr std::size_t n{10};
    constexpr auto sizes = make_sizes<initial_size,factor,n>();
    constexpr bool initialize_source = true;

    // SECTION("benchmark_memcpy_pageable_to_pageable"){
    //     std::cout<<std::endl<<std::endl<<"benchmark_memcpy_pageable_to_pageable"<<std::endl;
    //     for (const auto& size : sizes){
    //         auto host_buffer_src = pageable_buffer<value_type>(size);
    //         auto host_buffer_dst = pageable_buffer<value_type>(size);
    //         std::fill(host_buffer_src.begin(), host_buffer_src.end(),11.0f);
    //         auto copy_start = cpu_timer{};
    //         std::memcpy(host_buffer_dst.data(),host_buffer_src.data(),size*sizeof(value_type));
    //         auto copy_stop = cpu_timer{};
    //         auto dt_copy_ms = copy_stop - copy_start;
    //         std::cout<<std::endl<<size_to_str<value_type>(size)<<" "<<bandwidth_to_str<value_type>(size, dt_copy_ms)<<" copy timer "<<dt_copy_ms
    //             <<" src alignment "<<alignment(host_buffer_src.data())<<" dst alignment "<<alignment(host_buffer_dst.data());
    //     }
    // }
    // SECTION("benchmark_memcpy_pageable_to_pageable_unaligned"){
    //     std::cout<<std::endl<<std::endl<<"benchmark_memcpy_pageable_to_pageable_unaligned"<<std::endl;
    //     for (const auto& size : sizes){
    //         auto host_buffer_src = pageable_buffer<value_type>(size);
    //         auto host_buffer_dst = pageable_buffer<value_type>(size);
    //         std::fill(host_buffer_src.begin(), host_buffer_src.end(),11.0f);
    //         auto src_offset = 11;
    //         auto dst_offset = 3;
    //         auto new_size = size*sizeof(value_type)-src_offset - dst_offset;
    //         auto new_src = reinterpret_cast<unsigned char*>(host_buffer_src.data())+src_offset;
    //         auto new_dst = reinterpret_cast<unsigned char*>(host_buffer_dst.data())+dst_offset;
    //         auto copy_start = cpu_timer{};
    //         std::memcpy(new_dst, new_src,new_size);
    //         auto copy_stop = cpu_timer{};
    //         auto dt_copy_ms = copy_stop - copy_start;
    //         std::cout<<std::endl<<size_to_str<value_type>(size)<<" "<<bandwidth_to_str<value_type>(size, dt_copy_ms)<<" copy timer "<<dt_copy_ms
    //             <<" src alignment "<<alignment(new_src)<<" dst alignment "<<alignment(new_dst);
    //     }
    // }
    // SECTION("benchmark_memcpy_pageable_to_locked"){
    //     std::cout<<std::endl<<std::endl<<"benchmark_memcpy_pageable_to_locked"<<std::endl;
    //     for (const auto& size : sizes){
    //         auto host_buffer_src = pageable_buffer<value_type>(size);
    //         auto host_buffer_dst = locked_buffer<value_type>(size);
    //         std::fill(host_buffer_src.begin(), host_buffer_src.end(),111.0f);
    //         auto copy_start = cpu_timer{};
    //         std::memcpy(host_buffer_dst.data().get(),host_buffer_src.data(),size*sizeof(value_type));
    //         auto copy_stop = cpu_timer{};
    //         auto dt_copy_ms = copy_stop - copy_start;
    //         std::cout<<std::endl<<size_to_str<value_type>(size)<<" "<<bandwidth_to_str<value_type>(size, dt_copy_ms)<<" copy timer "<<dt_copy_ms
    //             <<" src alignment "<<alignment(host_buffer_src.data())<<" dst alignment "<<alignment(host_buffer_dst.data().get());
    //     }
    // }
    // SECTION("benchmark_memcpy_pageable_to_locked_unaligned"){
    //     std::cout<<std::endl<<std::endl<<"benchmark_memcpy_pageable_to_locked_unaligned"<<std::endl;
    //     for (const auto& size : sizes){
    //         auto host_buffer_src = pageable_buffer<value_type>(size);
    //         auto host_buffer_dst = locked_buffer<value_type>(size);
    //         std::fill(host_buffer_src.begin(), host_buffer_src.end(),111.0f);
    //         auto src_offset = 11;
    //         auto dst_offset = 3;
    //         auto new_size = size*sizeof(value_type)-src_offset - dst_offset;
    //         auto new_src = reinterpret_cast<unsigned char*>(host_buffer_src.data())+src_offset;
    //         auto new_dst = reinterpret_cast<unsigned char*>(host_buffer_dst.data().get())+dst_offset;
    //         auto copy_start = cpu_timer{};
    //         std::memcpy(new_dst,new_src,new_size);
    //         auto copy_stop = cpu_timer{};
    //         auto dt_copy_ms = copy_stop - copy_start;
    //         std::cout<<std::endl<<size_to_str<value_type>(size)<<" "<<bandwidth_to_str<value_type>(size, dt_copy_ms)<<" copy timer "<<dt_copy_ms
    //             <<" src alignment "<<alignment(new_src)<<" dst alignment "<<alignment(new_dst);
    //     }
    // }
}

// TEMPLATE_TEST_CASE("benchmark_experimental_avx_memcpy","[benchmark_experimental_copy]",
//     benchmark_experimental_copy::avx_memcpy_lu_su_test_helper,
//     benchmark_experimental_copy::avx_memcpy_lant_su_test_helper,
//     benchmark_experimental_copy::avx_memcpy_lu_sant_test_helper,
//     benchmark_experimental_copy::avx_memcpy_lant_sant_test_helper
// )
// {
//     using copier_type = TestType;
//     using value_type = float;
//     using cuda_experimental::cpu_timer;
//     using cuda_experimental::ptr_to_void;
//     using cuda_experimental::alignment;
//     using benchmark_helpers::make_sizes;
//     using benchmark_helpers::size_to_str;
//     using benchmark_helpers::bandwidth_to_str;
//     using cuda_experimental::locked_buffer;
//     using cuda_experimental::pageable_buffer;
//     using cuda_experimental::aligned_for_copy;


//     static_assert(std::is_trivially_copyable_v<value_type>);
//     constexpr std::size_t initial_size{1<<20};
//     constexpr std::size_t factor{2};
//     constexpr std::size_t n{10};
//     constexpr auto sizes = make_sizes<initial_size,factor,n>();
//     constexpr bool initialize_source = true;

//     std::cout<<std::endl<<std::endl<<"benchmark_avx_memcpy_pageable_to_locked"<<std::endl<<copier_type::name<<std::endl;
//     using block_type = __m256i;
//     for (const auto& size : sizes){
//         auto host_buffer_src = pageable_buffer<value_type>(size);
//         auto host_buffer_dst = locked_buffer<value_type>(size);
//         std::fill(host_buffer_src.begin(), host_buffer_src.end(),111.0f);
//         auto src_aligned = aligned_for_copy<block_type>(host_buffer_src.data(),host_buffer_src.size()*sizeof(value_type));
//         auto dst_aligned = aligned_for_copy<block_type>(host_buffer_dst.data().get(),host_buffer_dst.size()*sizeof(value_type));
//         auto n_copy = std::min(src_aligned.aligned_blocks(), dst_aligned.aligned_blocks())*src_aligned.block_size();
//         auto copy_start = cpu_timer{};
//         copier_type{}(dst_aligned.first_aligned(),src_aligned.first_aligned(),n_copy);
//         auto copy_stop = cpu_timer{};
//         auto dt_copy_ms = copy_stop - copy_start;
//         std::cout<<std::endl<<size_to_str<unsigned char>(n_copy)<<" "<<bandwidth_to_str<unsigned char>(n_copy, dt_copy_ms)<<" copy timer "<<dt_copy_ms
//             <<" src alignment "<<alignment(host_buffer_src.data())<<" dst alignment "<<alignment(host_buffer_dst.data().get());
//     }
// }

TEMPLATE_TEST_CASE("benchmark_experimental_memcpy_multithread","[benchmark_experimental_copy]",
    (std::integral_constant<std::size_t,0>),
    (std::integral_constant<std::size_t,1>),
    (std::integral_constant<std::size_t,2>),
    (std::integral_constant<std::size_t,3>),
    (std::integral_constant<std::size_t,4>),
    (std::integral_constant<std::size_t,5>),
    (std::integral_constant<std::size_t,6>),
    (std::integral_constant<std::size_t,7>),
    (std::integral_constant<std::size_t,8>)
)
{
    using threads_number_type = TestType;
    using value_type = float;
    using cuda_experimental::cpu_timer;
    using cuda_experimental::ptr_to_void;
    using cuda_experimental::alignment;
    using benchmark_helpers::make_sizes;
    using benchmark_helpers::size_to_str;
    using benchmark_helpers::bandwidth_to_str;
    using cuda_experimental::locked_buffer;
    using cuda_experimental::pageable_buffer;
    using cuda_experimental::aligned_for_copy;
    using benchmark_experimental_copy::memcpy_multithread;


    static_assert(std::is_trivially_copyable_v<value_type>);
    constexpr std::size_t initial_size{1<<20};
    constexpr std::size_t factor{2};
    constexpr std::size_t n{10};
    constexpr auto sizes = make_sizes<initial_size,factor,n>();
    constexpr bool initialize_source = true;

    std::cout<<std::endl<<std::endl<<"benchmark_memcpy_multithread_pageable_to_locked"<<" threads_number "<<threads_number_type{}()<<std::endl;
    for (const auto& size : sizes){
        auto host_buffer_src = pageable_buffer<value_type>(size);
        auto host_buffer_dst = locked_buffer<value_type>(size);
        std::fill(host_buffer_src.data(), host_buffer_src.data()+size,111.0f);
        auto copy_start = cpu_timer{};
        memcpy_multithread<threads_number_type{}()>(host_buffer_dst.data().get(),host_buffer_src.data(),size*sizeof(value_type));
        auto copy_stop = cpu_timer{};
        auto dt_copy_ms = copy_stop - copy_start;
        std::cout<<std::endl<<size_to_str<value_type>(size)<<" "<<bandwidth_to_str<value_type>(size, dt_copy_ms)<<" copy timer "<<dt_copy_ms
            <<" src alignment "<<alignment(host_buffer_src.data())<<" dst alignment "<<alignment(host_buffer_dst.data().get());
    }
}



// TEST_CASE("benchmark_cudaHostAlloc_cudaHostRegister","[benchmark_experimental_copy]"){

//     using benchmark_helpers::make_sizes;
//     using benchmark_helpers::size_to_str;
//     using cuda_experimental::cpu_timer;
//     using cuda_experimental::cuda_assert;
//     using value_type = float;

//     constexpr std::size_t initial_size{1<<20};
//     constexpr std::size_t factor{2};
//     constexpr std::size_t n{10};
//     constexpr auto sizes = make_sizes<initial_size,factor,n>();

//     std::cout<<std::endl<<std::endl<<"benchmark_cudaHostAlloc"<<std::endl;
//     for (auto size : sizes){
//         void *p;
//         auto alloc_start = cpu_timer{};
//         cuda_error_check(cudaHostAlloc(&p,size*sizeof(value_type),cudaHostAllocDefault));
//         auto free_start = cpu_timer{};
//         cuda_error_check(cudaFreeHost(p));
//         auto stop = cpu_timer{};
//         auto dt_alloc_ms = free_start - alloc_start;
//         auto dt_free_ms = stop - free_start;
//         std::cout<<std::endl<<size_to_str<value_type>(size)<<" alloc_timer "<<dt_alloc_ms<<" free_timer "<<dt_free_ms<<" total "<<dt_alloc_ms+dt_free_ms;
//     }

//     std::cout<<std::endl<<std::endl<<"benchmark_cudaHostRegister"<<std::endl;
//     for (auto size : sizes){
//         auto pageable_alloc_start = cpu_timer{};
//         auto p = new value_type[size];
//         auto register_start = cpu_timer{};
//         cuda_error_check(cudaHostRegister(p,size*sizeof(value_type),cudaHostRegisterDefault));
//         auto unregister_start = cpu_timer{};
//         cuda_error_check(cudaHostUnregister(p));
//         auto stop = cpu_timer{};
//         auto dt_pageable_alloc = register_start - pageable_alloc_start;
//         auto dt_register_ms = unregister_start - register_start;
//         auto dt_unregister_ms = stop - unregister_start;
//         std::cout<<std::endl<<size_to_str<value_type>(size)<<" pageable_alloc_timer "<<dt_pageable_alloc<<" register_timer "<<dt_register_ms<<" unregister_timer "<<dt_unregister_ms
//             <<" total "<<dt_pageable_alloc+dt_register_ms+dt_unregister_ms;
//     }

// }
