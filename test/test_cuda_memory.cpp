#include <iostream>
#include "catch.hpp"
#include "cuda_memory.hpp"


TEMPLATE_TEST_CASE("test_cuda_pointer","[test_cuda_memory]",
    cuda_experimental::cuda_pointer<float>,
    cuda_experimental::cuda_pointer<const float>
)
{
    using value_type = float;
    using cuda_pointer_type = TestType;

    REQUIRE(std::is_trivially_copyable_v<cuda_pointer_type>);
    float v{};
    auto p = cuda_pointer_type{};
    REQUIRE(p.get() == nullptr);
    SECTION("from_pointer_construction"){
        auto p = cuda_pointer_type{&v};
        REQUIRE(p.get() == &v);
        auto p1 = cuda_pointer_type{nullptr};
        REQUIRE(p1.get() == nullptr);
    }
    SECTION("copy_construction"){
        auto p = cuda_pointer_type{&v};
        auto p1 = p;
        REQUIRE(p1.get() == p.get());
    }
    SECTION("copy_assignment"){
        auto p = cuda_pointer_type{};
        auto p1 = cuda_pointer_type{&v};
        p = p1;
        REQUIRE(p.get() == p1.get());
        p1 = nullptr;
        REQUIRE(p1.get() == nullptr);
    }
    SECTION("equality"){
        auto p = cuda_pointer_type{};
        auto p1 = cuda_pointer_type{&v};
        auto p2 = cuda_pointer_type{&v};
        REQUIRE(p == p);
        REQUIRE(p1 == p1);
        REQUIRE(p1 == p2);
        REQUIRE(p1 != p);
    }
    SECTION("add_offset"){
        value_type a[10];
        auto p = cuda_pointer_type{a};
        REQUIRE(p+5 == cuda_pointer_type{a+5});
        REQUIRE(5+p == cuda_pointer_type{a+5});
    }
    SECTION("subtract_two_pointers"){
        value_type a[10];
        auto begin = cuda_pointer_type{a};
        auto end = cuda_pointer_type{a+10};
        REQUIRE(end-begin == 10);
        REQUIRE(end-end == 0);
        REQUIRE(distance(begin,end) == 10);
        REQUIRE(distance(begin,begin) == 0);
        REQUIRE(distance(begin,begin+3) == 3);
    }
    SECTION("distance"){
        value_type a[10];
        auto begin = cuda_pointer_type{a};
        auto end = cuda_pointer_type{a+10};
        REQUIRE(distance(begin,end) == 10);
        REQUIRE(distance(begin,begin) == 0);
        REQUIRE(distance(begin,begin+3) == 3);
        REQUIRE(distance(begin+2,end-3) == 5);
    }
}

TEST_CASE("test_copy","[test_cuda_memory]"){
    using value_type = float;
    using cuda_allocator_type = cuda_experimental::cuda_allocator<value_type>;
    using cuda_experimental::cuda_pointer;
    using cuda_experimental::copy;

    auto allocator = cuda_allocator_type{};
    constexpr std::size_t n{100};
    auto dev_ptr = allocator.allocate(n);
    auto const_dev_ptr = cuda_pointer<const value_type>{dev_ptr};

    SECTION("copy_host_device"){
        constexpr std::size_t a_len{10};
        value_type a[a_len] = {1,2,3,4,5,6,7,8,9,10};
        value_type a_copy[a_len]{};
        copy(a,a+a_len,dev_ptr);
        SECTION("copy_from_dev_ptr"){
            copy(dev_ptr,dev_ptr+a_len,a_copy);
            REQUIRE(std::equal(a,a+a_len,a_copy));
        }
        SECTION("copy_from_const_dev_ptr"){
            copy(const_dev_ptr,const_dev_ptr+a_len,a_copy);
            REQUIRE(std::equal(a,a+a_len,a_copy));
        }
    }
    SECTION("copy_host_device_iter"){
        auto a = std::vector<value_type>{1,2,3,4,5,6,7,8,9,10};
        auto a_len = a.size();
        auto a_copy = std::vector<value_type>(a_len);
        copy(a.begin(),a.end(),dev_ptr);
        SECTION("copy_from_dev_ptr"){
            copy(dev_ptr,dev_ptr+a_len,a_copy.begin());
            REQUIRE(std::equal(a.begin(),a.end(),a_copy.begin()));
        }
        SECTION("copy_from_const_dev_ptr"){
            copy(const_dev_ptr,const_dev_ptr+a_len,a_copy.begin());
            REQUIRE(std::equal(a.begin(),a.end(),a_copy.begin()));
        }
    }
    SECTION("copy_device_device"){
        constexpr std::size_t a_len{10};
        value_type a[a_len] = {1,2,3,4,5,6,7,8,9,10};
        value_type a_copy[a_len]{};
        copy(a,a+a_len,dev_ptr);
        auto dev_ptr_copy = allocator.allocate(a_len);
        SECTION("copy_from_dev_ptr"){
            copy(dev_ptr,dev_ptr+a_len,dev_ptr_copy);
            copy(dev_ptr_copy,dev_ptr_copy+a_len,a_copy);
            REQUIRE(std::equal(a,a+a_len,a_copy));
        }
        SECTION("copy_from_const_dev_ptr"){
            copy(const_dev_ptr,const_dev_ptr+a_len,dev_ptr_copy);
            copy(const_dev_ptr,const_dev_ptr+a_len,a_copy);
            REQUIRE(std::equal(a,a+a_len,a_copy));
        }
    }
    SECTION("copy_device_device_exception"){
        // auto p_first = cuda_pointer<value_type>{nullptr};
        // auto p_last = cuda_pointer<value_type>{nullptr};
        // REQUIRE_THROWS_AS(copy(p_first, p_last, dev_ptr),cuda_experimental::cuda_exception);
    }
    allocator.deallocate(dev_ptr,n);
}

TEST_CASE("test_cuda_pointer_attributes","[test_cuda_memory]"){
    using value_type = float;
    using cuda_mapping_allocator_type = cuda_experimental::cuda_mapping_allocator<value_type>;
    using cuda_allocator_type = cuda_experimental::cuda_allocator<value_type>;
    using cuda_experimental::cuda_pointer;
    using cuda_experimental::copy;
    using cuda_experimental::cuda_assert;
    using cuda_experimental::make_host_buffer;

// enum __device_builtin__ cudaMemoryType
// {
//     cudaMemoryTypeUnregistered = 0, /**< Unregistered memory */
//     cudaMemoryTypeHost         = 1, /**< Host memory */
//     cudaMemoryTypeDevice       = 2, /**< Device memory */
//     cudaMemoryTypeManaged      = 3  /**< Managed memory */
// };



    auto print_ptr_attr = [](const auto& p){
        cudaPointerAttributes attr;
        cuda_error_check(cudaPointerGetAttributes(&attr, p.get()));
        std::cout<<std::endl<<"device"<<attr.device;
        std::cout<<std::endl<<"device_ptr"<<attr.devicePointer;
        std::cout<<std::endl<<"host_ptr"<<attr.hostPointer;
        switch (attr.type){
            case cudaMemoryType::cudaMemoryTypeUnregistered:
                std::cout<<std::endl<<"Unregistered memory"<<attr.type;
                break;
            case cudaMemoryType::cudaMemoryTypeHost:
                std::cout<<std::endl<<"Host memory"<<attr.type;
                break;
            case cudaMemoryType::cudaMemoryTypeDevice:
                std::cout<<std::endl<<"Device memory"<<attr.type;
                break;
        }
    };

    int n{100};
    int offset{99};
    auto mapping_alloc = cuda_mapping_allocator_type{};
    auto p = mapping_alloc.allocate(n);
    print_ptr_attr(p+offset);

    auto buffer = make_host_buffer<value_type>(n);
    auto mapping_alloc_registered = cuda_mapping_allocator_type{buffer.get()};
    auto p_registered = mapping_alloc_registered.allocate(n);
    print_ptr_attr(p_registered+offset);

    auto dev_alloc = cuda_allocator_type{};
    auto p_dev = dev_alloc.allocate(n);
    print_ptr_attr(p_dev+offset);

    mapping_alloc.deallocate(p,n);
    mapping_alloc_registered.deallocate(p_registered,n);
    dev_alloc.deallocate(p_dev,n);
}