#include <numeric>
#include "catch.hpp"
#include "cuda_memory.hpp"
#include "benchmark_helpers.hpp"


TEST_CASE("test_memcpy_avx","[test_memcpy_avx]"){
    using cuda_experimental::cuda_memcpy::memcpy_avx;
    using cuda_experimental::align;
    using benchmark_helpers::make_sizes;
    using value_type = int;
    using host_allocator_type = std::allocator<value_type>;

    host_allocator_type host_alloc{};
    constexpr std::size_t initial_size{1000*1000};
    constexpr std::size_t factor{2};
    constexpr std::size_t n{5};
    constexpr auto sizes = make_sizes<initial_size,factor,n>();
    constexpr std::size_t block_alignment = alignof(cuda_experimental::cuda_memcpy::avx_block_type);
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
    using cuda_experimental::cuda_memcpy::uninitialized_copyn_multithread;
    using benchmark_helpers::make_sizes;
    constexpr std::size_t initial_size{1<<20};
    constexpr std::size_t factor{2};
    constexpr std::size_t n{10};
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

TEMPLATE_TEST_CASE("test_cuda_copy","[test_cuda_copy]", std::size_t)
{
    using value_type = typename TestType;
    using device_alloc_type = cuda_experimental::device_allocator<value_type>;
    using host_alloc_type = std::allocator<value_type>;
    using cuda_experimental::copy;
    using benchmark_helpers::make_sizes;
    // using pointer_type = typename allocator_type::pointer;
    // using const_pointer_type = typename allocator_type::const_pointer;

    constexpr std::size_t initial_size{1<<20};
    constexpr std::size_t factor{2};
    constexpr std::size_t n{10};
    constexpr auto sizes = make_sizes<initial_size,factor,n>();
    //{{1048576Ui64, 2097152Ui64, 4194304Ui64, 8388608Ui64, 16777216Ui64, 33554432Ui64, 67108864Ui64, 134217728Ui64, 268435456Ui64, 536870912Ui64}}
    device_alloc_type device_alloc{};
    host_alloc_type host_alloc{};

    SECTION("pointers_rage_src"){
        for (const auto& size : sizes){
            auto device_ptr = device_alloc.allocate(size);
            auto host_src_ptr = host_alloc.allocate(size);
            auto host_dst_ptr = host_alloc.allocate(size);
            std::iota(host_src_ptr, host_src_ptr+size, value_type{0});

            copy(host_src_ptr,host_src_ptr+size,device_ptr);
            copy(device_ptr,device_ptr+size,host_dst_ptr);

            REQUIRE(std::equal(host_src_ptr, host_src_ptr+size , host_dst_ptr));

            device_alloc.deallocate(device_ptr,size);
            host_alloc.deallocate(host_src_ptr,size);
            host_alloc.deallocate(host_dst_ptr,size);
        }
    }

    SECTION("iterastors_rage_src"){
        for (const auto& size : sizes){
            auto device_ptr = device_alloc.allocate(size);
            std::vector<value_type> host_src(size);
            std::vector<value_type> host_dst(size);
            std:iota(host_src.begin(), host_src.end(),value_type{0});

            copy(host_src.begin(),host_src.end(),device_ptr);
            copy(device_ptr,device_ptr+size,host_dst.begin());

            REQUIRE(std::equal(host_src.begin(), host_src.end() , host_dst.begin()));

            device_alloc.deallocate(device_ptr,size);
        }
    }


    // SECTION("copy_device_device"){
    //     value_type a_copy[a_len]{};
    //     copy(a,a+a_len,dev_ptr);
    //     auto dev_ptr_copy = allocator.allocate(a_len);
    //     SECTION("copy_from_dev_ptr"){
    //         copy(dev_ptr,dev_ptr+a_len,dev_ptr_copy);
    //         copy(dev_ptr_copy,dev_ptr_copy+a_len,a_copy);
    //     }
    //     SECTION("copy_from_const_dev_ptr"){
    //         copy(const_dev_ptr,const_dev_ptr+a_len,dev_ptr_copy);
    //         copy(const_dev_ptr,const_dev_ptr+a_len,a_copy);
    //     }
    //     REQUIRE(std::equal(a,a+a_len,a_copy));
    //     allocator.deallocate(dev_ptr_copy,a_len);
    // }
}

// TEMPLATE_TEST_CASE("test_copy","[test_cuda_memory]",
//     cuda_experimental::device_allocator<float>,
//     (cuda_experimental::device_allocator<test_cuda_memory::test_array<std::size_t,5>>)
// )
// {
//     using allocator_type = TestType;
//     using value_type = typename allocator_type::value_type;
//     using cuda_experimental::copy;
//     using pointer_type = typename allocator_type::pointer;
//     using const_pointer_type = typename allocator_type::const_pointer;

//     auto allocator = allocator_type{};
//     constexpr std::size_t n{100};
//     auto dev_ptr = allocator.allocate(n);
//     auto const_dev_ptr = const_pointer_type{dev_ptr};
//     constexpr std::size_t a_len{10};
//     value_type a[a_len] = {value_type(1),value_type(2),value_type(3),value_type(4),value_type(5),value_type(6),value_type(7),value_type(8),value_type(9),value_type(10)};

//     SECTION("copy_host_device"){
//         value_type a_copy[a_len]{};
//         copy(a,a+a_len,dev_ptr);
//         SECTION("copy_from_dev_ptr"){
//             copy(dev_ptr,dev_ptr+a_len,a_copy);
//         }
//         SECTION("copy_from_const_dev_ptr"){
//             copy(const_dev_ptr,const_dev_ptr+a_len,a_copy);
//         }
//         REQUIRE(std::equal(a,a+a_len,a_copy));
//     }
//     SECTION("copy_host_device_iter"){
//         auto v = std::vector<value_type>(a,a+a_len);
//         auto v_copy = std::vector<value_type>(a_len);
//         copy(v.begin(),v.end(),dev_ptr);
//         SECTION("copy_from_dev_ptr"){
//             copy(dev_ptr,dev_ptr+a_len,v_copy.begin());
//         }
//         SECTION("copy_from_const_dev_ptr"){
//             copy(const_dev_ptr,const_dev_ptr+a_len,v_copy.begin());
//         }
//         REQUIRE(std::equal(v.begin(),v.end(),v_copy.begin()));
//     }
//     SECTION("copy_device_device"){
//         value_type a_copy[a_len]{};
//         copy(a,a+a_len,dev_ptr);
//         auto dev_ptr_copy = allocator.allocate(a_len);
//         SECTION("copy_from_dev_ptr"){
//             copy(dev_ptr,dev_ptr+a_len,dev_ptr_copy);
//             copy(dev_ptr_copy,dev_ptr_copy+a_len,a_copy);
//         }
//         SECTION("copy_from_const_dev_ptr"){
//             copy(const_dev_ptr,const_dev_ptr+a_len,dev_ptr_copy);
//             copy(const_dev_ptr,const_dev_ptr+a_len,a_copy);
//         }
//         REQUIRE(std::equal(a,a+a_len,a_copy));
//         allocator.deallocate(dev_ptr_copy,a_len);
//     }
//     allocator.deallocate(dev_ptr,n);
// }