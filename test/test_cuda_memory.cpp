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
    auto v = float{};
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
        REQUIRE(p1.get() == &v);
    }
    SECTION("assignment"){
        auto p = cuda_pointer_type{};
        auto p1 = cuda_pointer_type{&v};
        p = p1;
        REQUIRE(p.get() == &v);
        REQUIRE(p1.get() == &v);
        p = nullptr;
        REQUIRE(p.get() == nullptr);
    }
    SECTION("equality"){
        auto p = cuda_pointer_type{};
        auto p1 = cuda_pointer_type{&v};
        auto p2 = cuda_pointer_type{&v};
        REQUIRE(p1 == p2);
        REQUIRE(p1 != p);
        REQUIRE(p2 != p);
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
    allocator.deallocate(dev_ptr,n);
}