#include <numeric>
#include <list>
#include <vector>
#include "catch.hpp"
#include "cuda_copy.hpp"
#include "benchmark_helpers.hpp"


TEST_CASE("test_memcpy_avx","[test_cuda_copy]"){
    using culib::cuda_copy::memcpy_avx;
    using culib::align;
    using benchmark_helpers::make_sizes;
    using value_type = int;
    using host_allocator_type = std::allocator<value_type>;

    host_allocator_type host_alloc{};
    constexpr std::size_t initial_size{1000*1000};
    constexpr std::size_t factor{2};
    constexpr std::size_t n_sizes{5};
    constexpr auto sizes = make_sizes<initial_size,factor,n_sizes>();
    constexpr std::size_t block_alignment = alignof(culib::cuda_copy::avx_block_type);
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
