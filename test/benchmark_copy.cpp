#include <array>
#include <sstream>
#include <iostream>
#include "catch.hpp"
#include "cuda_memory.hpp"
#include "benchmark_helpers.hpp"

TEST_CASE("test_benchmark_helpers","[benchmark_copy]"){

    using benchmark_helpers::make_size_helper;
    using benchmark_helpers::make_sizes;

    constexpr std::size_t initial_size{1<<20};
    constexpr std::size_t factor{2};
    constexpr auto(*make_size)(std::size_t) = &make_size_helper<initial_size,factor>;
    REQUIRE(make_size(0) == initial_size);
    REQUIRE(make_size(1) == factor*initial_size);
    REQUIRE(make_size(2) == factor*factor*initial_size);
    REQUIRE(make_size(3) == factor*factor*factor*initial_size);

    REQUIRE(make_sizes<initial_size,factor,1>() == std::array<std::size_t,1>{initial_size});
    REQUIRE(make_sizes<initial_size,factor,2>() == std::array<std::size_t,2>{initial_size, factor*initial_size});
    REQUIRE(make_sizes<initial_size,factor,3>() == std::array<std::size_t,3>{initial_size, factor*initial_size, factor*factor*initial_size});
    REQUIRE(make_sizes<initial_size,factor,4>() == std::array<std::size_t,4>{initial_size, factor*initial_size, factor*factor*initial_size, factor*factor*factor*initial_size});
}

TEMPLATE_TEST_CASE("benchmark_copy_host_device","[benchmark_copy]",
    //benchmark_helpers::pageable_uninitialized_buffer_maker<float>
    benchmark_helpers::pageable_initialized_buffer_maker<float>
    //benchmark_helpers::locked_buffer_maker<float>
    // benchmark_helpers::locked_write_combined_buffer_maker<float>
)
{
    using host_buffer_maker = TestType;
    using value_type = typename host_buffer_maker::value_type;
    using device_allocator_type = cuda_experimental::device_allocator<value_type>;
    using cuda_experimental::cuda_timer;
    using cuda_experimental::cpu_timer;
    using cuda_experimental::copy_locked;
    using cuda_experimental::cuda_assert;
    using cuda_experimental::copy_pageable;
    using benchmark_helpers::make_sizes;
    using benchmark_helpers::size_to_str;
    using benchmark_helpers::effective_bandwidth_to_str;
    using cuda_experimental::make_pageable_memory_buffer;
    using cuda_experimental::pageable_buffer;
    using cuda_experimental::locked_buffer;
    using cuda_experimental::locked_pointer;
    using cuda_experimental::locked_allocator;

    constexpr std::size_t initial_size{1<<20};
    constexpr std::size_t factor{2};
    constexpr std::size_t n{10};
    constexpr auto sizes = make_sizes<initial_size,factor,n>();
    constexpr auto max_size = sizes.back();
    device_allocator_type dev_allocator{};
    // //warming
    // for (std::size_t i{0}; i!=10; ++i){
    //     auto size = initial_size;
    //     auto host_buffer = host_buffer_maker{}(size);
    //     auto dev_buffer = dev_allocator.allocate(size);
    //     copy(host_buffer.begin(), host_buffer.end(),dev_buffer);
    //     dev_allocator.deallocate(dev_buffer,size);
    // }
    SECTION("benchmark_copy_host_to_device"){
        std::cout<<std::endl<<std::endl<<host_buffer_maker::name;
        std::cout<<std::endl<<"benchmark_copy_host_to_device"<<std::endl;
        auto buffer_allocator = locked_allocator<value_type>{};
        //benching host to dev
        for (const auto& size : sizes){
            auto host_buffer = host_buffer_maker{}(size);
            auto dev_buffer = dev_allocator.allocate(size);

            // auto copy_pageable_ = [&](auto pageable_first, auto pageable_last, auto d_first){
            //     auto n = std::distance(pageable_first,pageable_last);
            //     buffer_ = buffer_allocator.allocate(n);
            //     std::uninitialized_copy_n(pageable_first,n,buffer_.get());
            //     copy_locked(buffer_,buffer_+size,dev_buffer);
            // };
            // auto copy_pageable_ = [&](auto pageable_first, auto pageable_last, auto d_first){
            //     auto n = std::distance(pageable_first,pageable_last);
            //     auto buffer = locked_buffer<value_type>(n);
            //     std::uninitialized_copy_n(pageable_first,n,buffer.data().get());
            //     copy_locked(buffer.begin(),buffer.end(),dev_buffer);
            // };
            auto staging_start = cuda_timer{};
            auto buffer = buffer_allocator.allocate(size);
            std::uninitialized_copy_n(host_buffer.begin(),size,buffer.get());
            auto copy_start = cuda_timer{};
            auto dt_allocate = copy_start - staging_start;

            copy_locked(buffer,buffer+size,dev_buffer);

            auto staging_deallocate = cuda_timer{};
            auto dt_copy_ms = staging_deallocate - copy_start;

            auto cpu_deallocate_start = cpu_timer{};
            buffer_allocator.deallocate(buffer, size);
            auto cpu_deallocate_stop = cpu_timer{};
            auto stop = cuda_timer{};
            auto dt_deallocate_ms = stop - staging_deallocate;
            auto dt_cpu_deallocate = cpu_deallocate_stop - cpu_deallocate_start;
            std::cout<<std::endl<<size_to_str<value_type>(size)<<" "<<effective_bandwidth_to_str<value_type>(size, dt_copy_ms) << " allocate timer "<<dt_allocate
                <<" copy timer "<<dt_copy_ms<<" deallocate timer "<<dt_deallocate_ms<<" cpu_deallocate "<<dt_cpu_deallocate;
            dev_allocator.deallocate(dev_buffer, size);
        }
    }
    // SECTION("benchmark_copy_device_to_host"){
    //     std::cout<<std::endl<<std::endl<<host_buffer_maker::name;
    //     std::cout<<std::endl<<"benchmark_copy_device_to_host"<<std::endl;
    //     //warming
    //     for (std::size_t i{0}; i!=10; ++i){
    //         auto size = initial_size;
    //         auto host_buffer = host_buffer_maker{}(size);
    //         auto dev_buffer = dev_allocator.allocate(size);
    //         copy(dev_buffer, dev_buffer+size,host_buffer.begin());
    //         dev_allocator.deallocate(dev_buffer,size);
    //     }
    //     //benching dev to host
    //     for (const auto& size : sizes){
    //         auto host_buffer = host_buffer_maker{}(size);
    //         auto dev_buffer = dev_allocator.allocate(size);
    //         auto start = cuda_timer{};
    //         copy(dev_buffer, dev_buffer+size,host_buffer.begin());
    //         auto stop = cuda_timer{};
    //         auto dt_ms = stop - start;
    //         std::cout<<std::endl<<size_to_str<value_type>(size)<<" "<<effective_bandwidth_to_str<value_type>(size, dt_ms);
    //         dev_allocator.deallocate(dev_buffer,size);
    //     }
    // }
}

// TEST_CASE("benchmark_copy_same_device","[benchmark_copy]")
// {
//     using value_type = float;
//     using device_allocator_type = cuda_experimental::device_allocator<value_type>;
//     using cuda_experimental::cuda_timer;
//     using cuda_experimental::copy;
//     using benchmark_helpers::make_sizes;
//     using benchmark_helpers::size_to_str;
//     using benchmark_helpers::effective_bandwidth_to_str;

//     constexpr std::size_t initial_size{1<<20};
//     constexpr std::size_t factor{2};
//     constexpr std::size_t n{10};
//     constexpr auto sizes = make_sizes<initial_size,factor,n>();
//     constexpr auto max_size = sizes.back();
//     device_allocator_type dev_allocator{};
//     SECTION("benchmark_copy_device_to_device"){
//         std::cout<<std::endl<<std::endl<<"benchmark_copy_device_to_device"<<std::endl;
//         //warming
//         for (std::size_t i{0}; i!=10; ++i){
//             auto size = initial_size;
//             auto dev_buffer_src = dev_allocator.allocate(size);
//             auto dev_buffer_dst = dev_allocator.allocate(size);
//             copy(dev_buffer_src, dev_buffer_src+size,dev_buffer_dst);
//             dev_allocator.deallocate(dev_buffer_src,size);
//             dev_allocator.deallocate(dev_buffer_dst,size);
//         }
//         //benching dev to host
//         for (const auto& size : sizes){
//             auto dev_buffer_src = dev_allocator.allocate(size);
//             auto dev_buffer_dst = dev_allocator.allocate(size);
//             auto start = cuda_timer{};
//             copy(dev_buffer_src, dev_buffer_src+size,dev_buffer_dst);
//             auto stop = cuda_timer{};
//             auto dt_ms = stop - start;
//             std::cout<<std::endl<<size_to_str<value_type>(size)<<" "<<effective_bandwidth_to_str<value_type>(size, dt_ms);
//             dev_allocator.deallocate(dev_buffer_src,size);
//             dev_allocator.deallocate(dev_buffer_dst,size);
//         }
//     }
// }