#include <numeric>
#include <list>
#include <vector>
#include "catch.hpp"
#include "cuda_memory.hpp"
#include "benchmark_helpers.hpp"


TEST_CASE("test_memcpy_avx","[test_cuda_copy]"){
    using cuda_experimental::cuda_copy::memcpy_avx;
    using cuda_experimental::align;
    using benchmark_helpers::make_sizes;
    using value_type = int;
    using host_allocator_type = std::allocator<value_type>;

    host_allocator_type host_alloc{};
    constexpr std::size_t initial_size{1000*1000};
    constexpr std::size_t factor{2};
    constexpr std::size_t n{5};
    constexpr auto sizes = make_sizes<initial_size,factor,n>();
    constexpr std::size_t block_alignment = alignof(cuda_experimental::cuda_copy::avx_block_type);
    static_assert(block_alignment%sizeof(value_type) == 0);
    static_assert(block_alignment/sizeof(value_type) > 2);

    SECTION("dst_aligned_src_aligned"){
        for (const auto& size_ : sizes){
            auto src_ = host_alloc.allocate(size_);
            auto dst_ = host_alloc.allocate(size_);
            std::iota(src_, src_+size_, value_type{0});
            auto src = reinterpret_cast<value_type*>(align<block_alignment>(src_));
            auto dst = reinterpret_cast<value_type*>(align<block_alignment>(dst_));

            REQUIRE(align<block_alignment>(dst) == dst);
            REQUIRE(align<block_alignment>(src) == src);

            auto n = size_*sizeof(value_type) - 2*block_alignment;
            auto size = n/sizeof(value_type);
            memcpy_avx(dst, src, n);
            REQUIRE(std::equal(src, src+size, dst));
            host_alloc.deallocate(dst_,size_);
            host_alloc.deallocate(src_,size_);
        }
    }
    SECTION("dst_aligned_src_unaligned"){
        for (const auto& size_ : sizes){
            auto src_ = host_alloc.allocate(size_);
            auto dst_ = host_alloc.allocate(size_);
            std::iota(src_, src_+size_, value_type{0});
            auto src = reinterpret_cast<value_type*>(align<block_alignment>(src_));
            auto dst = reinterpret_cast<value_type*>(align<block_alignment>(dst_));

            ++src;
            REQUIRE(align<block_alignment>(dst) == dst);
            REQUIRE(align<block_alignment>(src) != src);

            auto n = size_*sizeof(value_type) - 2*block_alignment;
            auto size = n/sizeof(value_type);
            memcpy_avx(dst, src, n);
            REQUIRE(std::equal(src, src+size, dst));
            host_alloc.deallocate(dst_,size_);
            host_alloc.deallocate(src_,size_);
        }
    }
    SECTION("src_dst_unaligned_equal_offset"){
        for (const auto& size_ : sizes){
            auto src_ = host_alloc.allocate(size_);
            auto dst_ = host_alloc.allocate(size_);
            std::iota(src_, src_+size_, value_type{0});
            auto src = reinterpret_cast<value_type*>(align<block_alignment>(src_));
            auto dst = reinterpret_cast<value_type*>(align<block_alignment>(dst_));

            ++src;
            ++dst;
            REQUIRE(align<block_alignment>(dst) != dst);
            REQUIRE(align<block_alignment>(src) != src);
            REQUIRE(reinterpret_cast<std::uintptr_t>(src)%block_alignment == reinterpret_cast<std::uintptr_t>(dst)%block_alignment);

            auto n = size_*sizeof(value_type) - 2*block_alignment;
            auto size = n/sizeof(value_type);
            memcpy_avx(dst, src, n);
            REQUIRE(std::equal(src, src+size, dst));
            host_alloc.deallocate(dst_,size_);
            host_alloc.deallocate(src_,size_);
        }
    }
    SECTION("src_dst_unaligned_different_offset"){
        for (const auto& size_ : sizes){
            auto src_ = host_alloc.allocate(size_);
            auto dst_ = host_alloc.allocate(size_);
            std::iota(src_, src_+size_, value_type{0});
            auto src = reinterpret_cast<value_type*>(align<block_alignment>(src_));
            auto dst = reinterpret_cast<value_type*>(align<block_alignment>(dst_));

            ++src;
            ++dst;
            ++dst;
            REQUIRE(align<block_alignment>(dst) != dst);
            REQUIRE(align<block_alignment>(src) != src);
            REQUIRE(reinterpret_cast<std::uintptr_t>(src)%block_alignment != reinterpret_cast<std::uintptr_t>(dst)%block_alignment);

            auto n = size_*sizeof(value_type) - 2*block_alignment;
            auto size = n/sizeof(value_type);
            memcpy_avx(dst, src, n);
            REQUIRE(std::equal(src, src+size, dst));
            host_alloc.deallocate(dst_,size_);
            host_alloc.deallocate(src_,size_);
        }
    }
}

TEST_CASE("test_uninitialized_copyn_multithread", "[test_cuda_copy]"){
    using value_type = int;
    using host_alloc_type = std::allocator<value_type>;
    using cuda_experimental::cuda_copy::uninitialized_copyn_multithread;
    using benchmark_helpers::make_sizes;
    constexpr std::size_t initial_size{1<<10};
    constexpr std::size_t factor{2};
    constexpr std::size_t n{15};
    constexpr auto sizes = make_sizes<initial_size,factor,n>();
    constexpr std::size_t n_workers = 4;
    host_alloc_type alloc{};
    for (const auto& size : sizes){
        std::vector<value_type> src(size);
        std::iota(src.begin(),src.end(),value_type{0});
        auto dst = alloc.allocate(size);
        uninitialized_copyn_multithread<n_workers>(src.begin(),size,dst);
        REQUIRE(std::equal(src.begin(), src.end(), dst));
        alloc.deallocate(dst,size);
    }
}

TEMPLATE_TEST_CASE("test_cuda_copier_host_device_pointers_range","[test_cuda_copy]",
    (std::tuple<cuda_experimental::cuda_copy::copier<cuda_experimental::cuda_copy::native_copier_tag>,std::size_t>),
    (std::tuple<cuda_experimental::cuda_copy::copier<cuda_experimental::cuda_copy::multithread_copier_tag>,std::size_t>)
)
{
    using copier_type = std::tuple_element_t<0,TestType>;
    using value_type = std::tuple_element_t<1,TestType>;
    using device_alloc_type = cuda_experimental::device_allocator<value_type>;
    using host_alloc_type = std::allocator<value_type>;
    using benchmark_helpers::make_sizes;

    constexpr std::size_t initial_size{1<<10};
    constexpr std::size_t factor{2};
    constexpr std::size_t n{15};
    constexpr auto sizes = make_sizes<initial_size,factor,n>();
    device_alloc_type device_alloc{};
    host_alloc_type host_alloc{};
    using container_type = std::vector<value_type>;

    for (const auto& size : sizes){
        auto device_ptr = device_alloc.allocate(size);
        auto host_src_ptr = host_alloc.allocate(size);
        auto host_dst_ptr = host_alloc.allocate(size);
        std::iota(host_src_ptr, host_src_ptr+size, value_type{0});
        auto res_hd = copier_type::copy(host_src_ptr,host_src_ptr+size,device_ptr);
        auto res_dh = copier_type::copy(device_ptr,device_ptr+size,host_dst_ptr);
        REQUIRE(res_hd == device_ptr+size);
        REQUIRE(res_dh == host_dst_ptr+size);
        REQUIRE(std::equal(host_src_ptr, host_src_ptr+size , host_dst_ptr));
        device_alloc.deallocate(device_ptr,size);
        host_alloc.deallocate(host_src_ptr,size);
        host_alloc.deallocate(host_dst_ptr,size);
    }
}

TEMPLATE_TEST_CASE("test_cuda_copier_host_device_iterators_range","[test_cuda_copy]",
    (std::tuple<cuda_experimental::cuda_copy::copier<cuda_experimental::cuda_copy::native_copier_tag>,std::vector<std::size_t>>),
    (std::tuple<cuda_experimental::cuda_copy::copier<cuda_experimental::cuda_copy::native_copier_tag>,std::list<std::size_t>>),
    (std::tuple<cuda_experimental::cuda_copy::copier<cuda_experimental::cuda_copy::multithread_copier_tag>,std::vector<std::size_t>>),
    (std::tuple<cuda_experimental::cuda_copy::copier<cuda_experimental::cuda_copy::multithread_copier_tag>,std::list<std::size_t>>)
)
{
    using copier_type = std::tuple_element_t<0,TestType>;
    using container_type = std::tuple_element_t<1,TestType>;
    using value_type = typename container_type::value_type;
    using device_alloc_type = cuda_experimental::device_allocator<value_type>;
    using host_alloc_type = std::allocator<value_type>;
    using benchmark_helpers::make_sizes;

    constexpr std::size_t initial_size{1<<10};
    constexpr std::size_t factor{2};
    constexpr std::size_t n{10};
    constexpr auto sizes = make_sizes<initial_size,factor,n>();
    device_alloc_type device_alloc{};
    host_alloc_type host_alloc{};

    for (const auto& size : sizes){
        auto device_ptr = device_alloc.allocate(size);
        container_type host_src(size);
        container_type host_dst(size);
        std:iota(host_src.begin(), host_src.end(),value_type{0});
        auto res_hd = copier_type::copy(host_src.begin(),host_src.end(),device_ptr);
        auto res_dh = copier_type::copy(device_ptr,device_ptr+size,host_dst.begin());
        REQUIRE(res_hd == device_ptr+size);
        REQUIRE(res_dh == host_dst.end());
        REQUIRE(std::equal(host_src.begin(), host_src.end() , host_dst.begin()));
        device_alloc.deallocate(device_ptr,size);
    }
}

TEMPLATE_TEST_CASE("test_cuda_copier_device_device","[test_cuda_copy]",
    (std::tuple<cuda_experimental::cuda_copy::copier<cuda_experimental::cuda_copy::native_copier_tag>,std::size_t>),
    (std::tuple<cuda_experimental::cuda_copy::copier<cuda_experimental::cuda_copy::multithread_copier_tag>,std::size_t>)
)
{
    using copier_type = std::tuple_element_t<0,TestType>;
    using value_type = std::tuple_element_t<1,TestType>;
    using device_alloc_type = cuda_experimental::device_allocator<value_type>;
    using host_alloc_type = std::allocator<value_type>;
    using benchmark_helpers::make_sizes;

    constexpr std::size_t initial_size{1<<10};
    constexpr std::size_t factor{2};
    constexpr std::size_t n{15};
    constexpr auto sizes = make_sizes<initial_size,factor,n>();
    device_alloc_type device_alloc{};
    host_alloc_type host_alloc{};

    SECTION("copy_same_device"){
        for (const auto& size : sizes){
            auto host_src = host_alloc.allocate(size);
            auto host_dst = host_alloc.allocate(size);
            auto device0_src = device_alloc.allocate(size);
            auto device0_dst = device_alloc.allocate(size);
            std::iota(host_src, host_src+size,value_type{0});
            //host_src -> device0_src -> device0_dst -> host_dst
            auto res_hd = copier_type::copy(host_src,host_src+size,device0_src);
            auto res_dd = copier_type::copy(device0_src,device0_src+size,device0_dst);
            auto res_dh = copier_type::copy(device0_dst,device0_dst+size,host_dst);

            REQUIRE(res_hd == device0_src+size);
            REQUIRE(res_dd == device0_dst+size);
            REQUIRE(res_dh == host_dst+size);
            REQUIRE(std::equal(host_src, host_src+size, host_dst));

            host_alloc.deallocate(host_src,size);
            host_alloc.deallocate(host_dst,size);
            device_alloc.deallocate(device0_src,size);
            device_alloc.deallocate(device0_dst,size);
        }
    }

    SECTION("copy_peer_device"){
        using cuda_experimental::cuda_get_device_count;
        using cuda_experimental::cuda_get_device;
        using cuda_experimental::cuda_set_device;
        if (cuda_get_device_count() > 1){
            constexpr int device0_id = 0;
            constexpr int device1_id = 1;
            for (const auto& size : sizes){
                auto host_src = host_alloc.allocate(size);
                auto host_dst = host_alloc.allocate(size);
                std::iota(host_src, host_src+size,value_type{0});
                cuda_set_device(device0_id);
                auto device0_src = device_alloc.allocate(size);
                cuda_set_device(device1_id);
                auto device1_dst = device_alloc.allocate(size);
                //host_src -> device0_src -> device1_dst -> host_dst
                auto res_hd = copier_type::copy(host_src,host_src+size,device0_src);
                auto res_dd = copier_type::copy(device0_src,device0_src+size,device1_dst);
                auto res_dh = copier_type::copy(device1_dst,device1_dst+size,host_dst);

                REQUIRE(res_hd == device0_src+size);
                REQUIRE(res_dd == device1_dst+size);
                REQUIRE(res_dh == host_dst+size);
                REQUIRE(std::equal(host_src, host_src+size, host_dst));

                host_alloc.deallocate(host_src,size);
                host_alloc.deallocate(host_dst,size);
                device_alloc.deallocate(device0_src,size);
                device_alloc.deallocate(device1_dst,size);
            }
        }
    }
}

TEMPLATE_TEST_CASE("test_cuda_fill", "[test_cuda_copy]",
    float
)
{
    using value_type = TestType;
    using device_alloc_type = cuda_experimental::device_allocator<value_type>;
    using host_alloc_type = std::allocator<value_type>;
    using benchmark_helpers::make_sizes;
    using cuda_experimental::fill;
    using cuda_experimental::copy;

    constexpr std::size_t initial_size{1<<10};
    constexpr std::size_t factor{2};
    constexpr std::size_t n{15};
    constexpr auto sizes = make_sizes<initial_size,factor,n>();
    device_alloc_type device_alloc{};
    host_alloc_type host_alloc{};
    value_type v{11.0f};

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