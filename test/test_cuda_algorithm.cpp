#include <numeric>
#include <list>
#include <vector>
#include "catch.hpp"
#include "cuda_algorithm.hpp"
#include "benchmark_helpers.hpp"


TEST_CASE("test_cuda_fill", "[test_cuda_algorithm]")
{
    using value_type = double;
    using device_alloc_type = culib::device_allocator<value_type>;
    using host_alloc_type = std::allocator<value_type>;
    using benchmark_helpers::make_sizes;
    using culib::fill;
    using culib::copy;

    constexpr std::size_t initial_size{1<<10};
    constexpr std::size_t factor{2};
    constexpr std::size_t n{15};
    constexpr auto sizes = make_sizes<initial_size,factor,n>();
    device_alloc_type device_alloc{};
    host_alloc_type host_alloc{};
    value_type v{11.0};

    for (const auto& size : sizes){
        auto host_ptr = host_alloc.allocate(size);
        auto device_ptr = device_alloc.allocate(size);
        std::vector<value_type> expected(size, v);
        fill(device_ptr, device_ptr+size, v);
        copy(device_ptr, device_ptr+size, host_ptr);
        REQUIRE(std::equal(host_ptr, host_ptr+size, expected.begin()));
        host_alloc.deallocate(host_ptr,size);
        device_alloc.deallocate(device_ptr,size);
    }
}

TEST_CASE("test_cuda_copy", "[test_cuda_algorithm]")
{
    using value_type = double;
    using device_alloc_type = culib::device_allocator<value_type>;
    using benchmark_helpers::make_sizes;
    using culib::copy;

    device_alloc_type device_alloc{};
    const std::size_t n = 64*1024*1024;
    std::vector<value_type> host_src(n);
    std::vector<value_type> host_dst(n,0);
    std::iota(host_src.begin(),host_src.end(),value_type(0));
    auto device_first = device_alloc.allocate(n);
    auto device_last = device_first + n;

    SECTION("pointer")
    {
        copy(host_src.data(),host_src.data()+n,device_first);
        copy(device_first,device_last,host_dst.data());
        REQUIRE(std::equal(host_src.begin(),host_src.end(),host_dst.begin()));
    }
    SECTION("iterator")
    {
        copy(host_src.begin(),host_src.end(),device_first);
        copy(device_first,device_last,host_dst.begin());
        REQUIRE(std::equal(host_src.begin(),host_src.end(),host_dst.begin()));
    }
    device_alloc.deallocate(device_first,n);
}