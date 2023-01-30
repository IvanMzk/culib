#include <iostream>
#include <array>
#include <numeric>
#include "catch.hpp"
#include "cuda_memory.hpp"

namespace test_cuda_memory{
using cuda_experimental::basic_pointer;
template<typename T>
class test_pointer : public basic_pointer<T, test_pointer>
{
public:
    using typename basic_pointer::pointer;
    using typename basic_pointer::value_type;
    //operator test_pointer<const T>()const{return test_pointer<const T>{get()};}
    using basic_pointer::operator=;
    explicit test_pointer(pointer p = nullptr):
        basic_pointer(p)
    {}
};
}   //end of namespace test_cuda_memory

TEMPLATE_TEST_CASE("test_basic_pointer","[test_cuda_memory]",
    test_cuda_memory::test_pointer<float>,
    test_cuda_memory::test_pointer<const float>
)
{
    using value_type = float;
    using pointer_type = TestType;

    REQUIRE(std::is_trivially_copyable_v<pointer_type>);
    value_type v{};
    auto p = pointer_type{};
    REQUIRE(p.get() == nullptr);
    SECTION("from_pointer_construction"){
        auto p = pointer_type{&v};
        REQUIRE(p.get() == &v);
        auto p1 = pointer_type{nullptr};
        REQUIRE(p1.get() == nullptr);
    }
    SECTION("copy_construction"){
        auto p = pointer_type{&v};
        auto p1 = p;
        REQUIRE(p1.get() == p.get());
    }
    SECTION("copy_assignment"){
        auto p = pointer_type{};
        auto p1 = pointer_type{&v};
        p = p1;
        REQUIRE(p.get() == p1.get());
        p1 = nullptr;
        REQUIRE(p1.get() == nullptr);
    }
    SECTION("equality"){
        auto p = pointer_type{};
        auto p1 = pointer_type{&v};
        auto p2 = pointer_type{&v};
        REQUIRE(p == p);
        REQUIRE(p1 == p1);
        REQUIRE(p1 == p2);
        REQUIRE(p1 != p);
    }
    SECTION("add_offset"){
        value_type a[10];
        auto p = pointer_type{a};
        REQUIRE(std::is_same_v<decltype(p+5),pointer_type>);
        REQUIRE(std::is_same_v<decltype(5+p),pointer_type>);
        REQUIRE(p+5 == pointer_type{a+5});
        REQUIRE(5+p == pointer_type{a+5});
    }
    SECTION("subtract_two_pointers"){
        value_type a[10];
        auto begin = pointer_type{a};
        auto end = pointer_type{a+10};
        REQUIRE(end-begin == 10);
        REQUIRE(end-end == 0);
        REQUIRE(distance(begin,end) == 10);
        REQUIRE(distance(begin,begin) == 0);
        REQUIRE(distance(begin,begin+3) == 3);
    }
    SECTION("distance"){
        value_type a[10];
        auto begin = pointer_type{a};
        auto end = pointer_type{a+10};
        REQUIRE(distance(begin,end) == 10);
        REQUIRE(distance(begin,begin) == 0);
        REQUIRE(distance(begin,begin+3) == 3);
        REQUIRE(distance(begin+2,end-3) == 5);
    }
    SECTION("bool_cast_operator"){
        pointer_type p{};
        REQUIRE(!static_cast<bool>(p));
        pointer_type p1{&v};
        REQUIRE(static_cast<bool>(p1));
    }
    SECTION("mutating_addition_operators"){
        value_type a[10];
        auto p = pointer_type{a};
        REQUIRE(++p == pointer_type{a+1});
        REQUIRE(p++ == pointer_type{a+1});
        REQUIRE(p == pointer_type{a+2});
        REQUIRE((p+=3) == pointer_type{a+5});
    }
    SECTION("mutating_subtraction_operators"){
        constexpr std::size_t n{10};
        value_type a[n];
        auto p = pointer_type{a+n};
        REQUIRE(--p == pointer_type{a+n-1});
        REQUIRE(p-- == pointer_type{a+n-1});
        REQUIRE(p == pointer_type{a+n-2});
        REQUIRE((p-=3) == pointer_type{a+n-5});
    }
    SECTION("comparisons"){
        constexpr std::size_t n{10};
        value_type a[n];
        auto p = pointer_type{a};
        REQUIRE(p > pointer_type{});
        REQUIRE(pointer_type{} < p);
        REQUIRE(pointer_type{a+1} > p);
        REQUIRE(p < pointer_type{a+1});
        REQUIRE(p <= p);
        REQUIRE(p >= p);
        REQUIRE(pointer_type{} >= pointer_type{});
        REQUIRE(pointer_type{} <= pointer_type{});
        REQUIRE(pointer_type{a+1} >= p);
        REQUIRE(p <= pointer_type{a+1});
    }
}

TEMPLATE_TEST_CASE("test_device_pointer","[test_cuda_memory]",
    float
)
{
    using value_type = TestType;
    using device_allocator_type = cuda_experimental::device_allocator<value_type>;
    using cuda_experimental::cuda_get_device;
    using cuda_experimental::device_pointer;

    device_allocator_type allocator{};
    constexpr size_t n = 100;
    std::vector<value_type> expected(n);
    std:iota(expected.begin(), expected.end(), value_type{0});
    auto device = cuda_get_device();
    auto ptr_dev = allocator.allocate(n);
    REQUIRE(ptr_dev.device() == device);
    copy(expected.begin(), expected.end(), ptr_dev);

    SECTION("device_pointer_asignment"){
        device_pointer<value_type> p{};
        p = ptr_dev;
        REQUIRE(p.get() == ptr_dev.get());
        REQUIRE(p.device() == ptr_dev.device());
    }

    SECTION("device_pointer_equality"){
        device_pointer<value_type> p{};
        REQUIRE(p != ptr_dev);
        device_pointer<value_type> p1{ptr_dev.get(),ptr_dev.device()+1};
        REQUIRE(p1 != ptr_dev);
        device_pointer<value_type> p2{ptr_dev.get(),ptr_dev.device()};
        REQUIRE(p2 == ptr_dev);
        p = ptr_dev;
        REQUIRE(p == ptr_dev);
    }

    SECTION("device_pointer_dereference_read"){
        std::vector<value_type> result{};
        std::for_each_n(ptr_dev,n,[&result](const auto& v){result.push_back(v);});
        REQUIRE(std::equal(expected.begin(),expected.end(),result.begin()));
    }
    SECTION("device_pointer_to_const_dereference_read"){
        std::vector<value_type> result{};
        std::for_each_n(device_pointer<const value_type>{ptr_dev.get(), ptr_dev.device()},n,[&result](const auto& v){result.push_back(v);});
        REQUIRE(std::equal(expected.begin(),expected.end(),result.begin()));
    }
    SECTION("device_pointer_dereference_write"){
        auto dev_dst = allocator.allocate(n);
        auto expected_it = expected.begin();
        for(auto it=dev_dst, end=it+n; it!=end; ++it, ++expected_it){
            *it = *expected_it;
        }
        std::vector<value_type> result{};
        std::for_each_n(dev_dst,n,[&result](const auto& v){result.push_back(v);});
        REQUIRE(std::equal(expected.begin(),expected.end(),result.begin()));
        allocator.deallocate(dev_dst, n);
    }
    SECTION("device_pointer_subscription"){
        auto dev_dst = allocator.allocate(n);
        for(std::size_t i{0}; i!=n; ++i){
            dev_dst[i] = expected[i];
        }
        std::vector<value_type> result{};
        for (std::size_t i{0}; i!=n; ++i){
            result.push_back(dev_dst[i]);
        }
        REQUIRE(std::equal(expected.begin(),expected.end(),result.begin()));
        allocator.deallocate(dev_dst, n);
    }
    SECTION("device_pointer_with_algorithm"){
        auto transformator = [](const auto& v){return v+1;};
        std::vector<value_type> expected_transformed(n);
        std::transform(expected.begin(),expected.end(),expected_transformed.begin(),transformator);
        auto dev_transformed = allocator.allocate(n);
        std::transform(ptr_dev,ptr_dev+n,dev_transformed,transformator);
        REQUIRE(std::equal(dev_transformed, dev_transformed+n, expected_transformed.begin()));
        allocator.deallocate(dev_transformed, n);
    }
    allocator.deallocate(ptr_dev, n);
}
