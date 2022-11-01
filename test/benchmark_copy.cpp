#include <array>
#include <sstream>
#include <iostream>
#include "catch.hpp"
#include "cuda_memory.hpp"

namespace benchmark_copy{

    template<std::size_t Init, std::size_t Fact>
    constexpr auto make_size_helper(std::size_t i){
        if (i==0){
            return Init;
        }else{
            return Fact*make_size_helper<Init,Fact>(i-1);
        }
    }
    template<std::size_t Init, std::size_t Fact, std::size_t...I>
    auto constexpr make_sizes_helper(std::index_sequence<I...>){
        return std::array<std::size_t, sizeof...(I)>{make_size_helper<Init,Fact>(I)...};
    }
    template<std::size_t Init = (1<<20), std::size_t Fact = 2, std::size_t N>
    auto constexpr make_sizes(){
        return make_sizes_helper<Init,Fact>(std::make_index_sequence<N>{});
    }

    template<typename T>
    auto size_in_bytes(std::size_t n){return n*sizeof(T);}
    template<typename T>
    auto size_in_mbytes(std::size_t n){return size_in_bytes<T>(n)/std::size_t{1000000};}
    template<typename T>
    auto size_in_gbytes(std::size_t n){return size_in_bytes<T>(n)/std::size_t{1000000000};}

    template<typename T>
    auto size_to_str(std::size_t n){
        std::stringstream ss{};
        ss<<size_in_mbytes<T>(n)<<"MByte";
        return ss.str();
    }
    template<typename T>
    auto effective_bandwidth_to_str(std::size_t n, float dt_ms){
        std::stringstream ss{};
        ss<<size_in_mbytes<T>(n)/dt_ms<<"GBytes/s";
        return ss.str();
    }

    template<typename T>
    struct pageable_buffer_maker
    {
        using value_type = T;
        static constexpr char name[] = "pageable_buffer_maker";
        template<typename U>
        auto operator()(const U& n){return cuda_experimental::pageable_buffer<value_type>(n);}
    };
    template<typename T>
    struct locked_buffer_maker
    {
        using value_type = T;
        static constexpr char name[] = "locked_buffer_maker";
        template<typename U>
        auto operator()(const U& n){return cuda_experimental::locked_buffer<value_type>(n);}
    };
    // template<typename T>
    // struct locked_write_combined_buffer_maker
    // {
    //     using value_type = T;
    //     static constexpr char name[] = "locked_write_combined_buffer_maker";
    //     template<typename U>
    //     auto operator()(const U& n){return cuda_experimental::make_locked_memory_buffer<value_type>(n,cudaHostAllocWriteCombined);}
    // };

}


TEST_CASE("test_benchmark_copy_helpers","[benchmark_copy]"){

    using benchmark_copy::make_size_helper;
    using benchmark_copy::make_sizes;

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
    benchmark_copy::pageable_buffer_maker<float>
    //benchmark_copy::locked_buffer_maker<float>
    // benchmark_copy::locked_write_combined_buffer_maker<float>
)
{
    using host_buffer_maker = TestType;
    using value_type = typename host_buffer_maker::value_type;
    using device_allocator_type = cuda_experimental::device_allocator<value_type>;
    using cuda_experimental::cuda_timer;
    using cuda_experimental::copy;
    using benchmark_copy::make_sizes;
    using benchmark_copy::size_to_str;
    using benchmark_copy::effective_bandwidth_to_str;
    using cuda_experimental::make_pageable_memory_buffer;
    using cuda_experimental::pageable_buffer;

    constexpr std::size_t initial_size{1<<20};
    constexpr std::size_t factor{2};
    constexpr std::size_t n{10};
    constexpr auto sizes = make_sizes<initial_size,factor,n>();
    constexpr auto max_size = sizes.back();
    device_allocator_type dev_allocator{};
    SECTION("benchmark_copy_host_to_device"){
        std::cout<<std::endl<<std::endl<<host_buffer_maker::name;
        std::cout<<std::endl<<"benchmark_copy_host_to_device"<<std::endl;
        //warming
        for (std::size_t i{0}; i!=10; ++i){
            auto size = initial_size;
            auto host_buffer = host_buffer_maker{}(size);
            auto dev_buffer = dev_allocator.allocate(size);
            copy(host_buffer.begin(), host_buffer.end(),dev_buffer);
            dev_allocator.deallocate(dev_buffer,size);
        }
        //benching host to dev
        for (const auto& size : sizes){
            auto host_buffer = host_buffer_maker{}(size);
            auto dev_buffer = dev_allocator.allocate(size);
            auto start = cuda_timer{};
            copy(host_buffer.begin(), host_buffer.end(),dev_buffer);
            auto stop = cuda_timer{};
            auto dt_ms = stop - start;
            std::cout<<std::endl<<size_to_str<value_type>(size)<<" "<<effective_bandwidth_to_str<value_type>(size, dt_ms);
            dev_allocator.deallocate(dev_buffer, size);
        }
    }
    SECTION("benchmark_copy_device_to_host"){
        std::cout<<std::endl<<std::endl<<host_buffer_maker::name;
        std::cout<<std::endl<<"benchmark_copy_device_to_host"<<std::endl;
        //warming
        for (std::size_t i{0}; i!=10; ++i){
            auto size = initial_size;
            auto host_buffer = host_buffer_maker{}(size);
            auto dev_buffer = dev_allocator.allocate(size);
            copy(dev_buffer, dev_buffer+size,host_buffer.begin());
            dev_allocator.deallocate(dev_buffer,size);
        }
        //benching dev to host
        for (const auto& size : sizes){
            auto host_buffer = host_buffer_maker{}(size);
            auto dev_buffer = dev_allocator.allocate(size);
            auto start = cuda_timer{};
            copy(dev_buffer, dev_buffer+size,host_buffer.begin());
            auto stop = cuda_timer{};
            auto dt_ms = stop - start;
            std::cout<<std::endl<<size_to_str<value_type>(size)<<" "<<effective_bandwidth_to_str<value_type>(size, dt_ms);
            dev_allocator.deallocate(dev_buffer,size);
        }
    }
}

// TEST_CASE("benchmark_copy_same_device","[benchmark_copy]")
// {
//     using value_type = float;
//     using device_allocator_type = cuda_experimental::device_allocator<value_type>;
//     using cuda_experimental::cuda_timer;
//     using cuda_experimental::copy;
//     using benchmark_copy::make_sizes;
//     using benchmark_copy::size_to_str;
//     using benchmark_copy::effective_bandwidth_to_str;

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