#include <iostream>
#include <array>
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
    operator test_pointer<const T>()const{return test_pointer<const T>{get()};}
    using basic_pointer::operator=;
    explicit test_pointer(pointer p = nullptr):
        basic_pointer(p)
    {}
};

template<typename T, std::size_t N>
class test_array
{
    T elements_[N]{};
public:
    test_array() = default;
    explicit test_array(const T& v){
        std::fill_n(elements_,N,v);
    }
    constexpr std::size_t size()const{return N;}
    bool operator==(const test_array& other){return std::equal(elements_,elements_+N, other.elements_);}
};

template<typename T>
class test_lin_space
{
    T min_;
    T max_;
    std::size_t points_number_;
public:
    test_lin_space(const T& min__, const T& max__, std::size_t points_number__):
        min_{min__},
        max_{max__},
        points_number_{points_number__}
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
    SECTION("cast_to_pointer_to_const_operator"){
        using test_cuda_memory::test_pointer;
        test_pointer<value_type> p{&v};
        //auto cp = static_cast<test_pointer<const value_type>>(p);
        //REQUIRE(p == cp);
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

TEST_CASE("test_device_pointer","[test_cuda_memory]"){
    using value_type = float;
    using allocator_type = cuda_experimental::device_allocator<value_type>;
    using cuda_experimental::cuda_get_device;
    using cuda_experimental::device_pointer;

    allocator_type allocator{};
    std::vector<value_type> v{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    auto n = v.size();
    auto device = cuda_get_device();
    auto ptr_dev = allocator.allocate(n);
    REQUIRE(ptr_dev.device() == device);
    copy(v.begin(), v.end(), ptr_dev);

    auto cp = static_cast<device_pointer<const value_type>>(ptr_dev);

    SECTION("device_pointer_dereference"){
        auto it = ptr_dev;
        auto end = ptr_dev+n;
        auto const_it = ptr_to_const(it);
        SECTION("read_dev_reference"){
            std::vector<value_type> v_dev_copy{};
            std::vector<value_type> v_const_dev_copy{};
            for(;it!=end; ++it, ++const_it){
                v_dev_copy.push_back(*it);
                v_const_dev_copy.push_back(*const_it);
            }
            REQUIRE(std::equal(v.begin(),v.end(),v_dev_copy.begin()));
            REQUIRE(std::equal(v.begin(),v.end(),v_const_dev_copy.begin()));
        }
        SECTION("write_dev_reference"){
            std::vector<value_type> v_expected_result{2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40};
            std::vector<value_type> v_expected_result_{};
            for(;it!=end; ++it){
                auto v = *it*2;
                *it = v;
            }
            std::vector<value_type> v_dev_copy(n);
            copy(ptr_dev, ptr_dev+n, v_dev_copy.begin());
            REQUIRE(std::equal(v_expected_result.begin(),v_expected_result.end(),v_dev_copy.begin()));
        }
    }
    SECTION("device_pointer_subscription"){
        SECTION("read_dev_reference"){
            std::vector<value_type> v_dev_copy{};
            std::vector<value_type> v_const_dev_copy{};
            auto ptr_const_dev = ptr_to_const(ptr_dev);
            for(std::size_t i{0};i!=n; ++i){
                v_dev_copy.push_back(ptr_dev[i]);
                v_const_dev_copy.push_back(ptr_const_dev[i]);
            }
            REQUIRE(std::equal(v.begin(),v.end(),v_dev_copy.begin()));
            REQUIRE(std::equal(v.begin(),v.end(),v_const_dev_copy.begin()));
        }
        SECTION("write_dev_reference"){
            std::vector<value_type> v_expected_result{};
            for(std::size_t i{0}; i!=n; ++i){
                auto v = i%2;
                ptr_dev[i] = v;
                v_expected_result.push_back(v);
            }
            std::vector<value_type> v_dev_copy(n);
            copy(ptr_dev, ptr_dev+n, v_dev_copy.begin());
            REQUIRE(std::equal(v_expected_result.begin(),v_expected_result.end(),v_dev_copy.begin()));
        }
    }
    SECTION("device_pointer_iteration"){
        auto begin = ptr_dev;
        auto end = ptr_dev+n;
        SECTION("read"){
            auto cbegin = ptr_to_const(begin);
            auto cend = ptr_to_const(end);
            REQUIRE(std::equal(begin, end, v.begin()));
            REQUIRE(std::equal(cbegin, cend, v.begin()));
        }
        SECTION("write"){
            auto transformed_begin = allocator.allocate(n);
            auto transformator = [](const auto& v){return v+1;};
            std::vector<value_type> v_transformed{2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21};
            std::transform(begin,end,transformed_begin,transformator);

            REQUIRE(std::equal(v_transformed.begin(),v_transformed.end(), transformed_begin));
            allocator.deallocate(transformed_begin,n);
        }
    }

    allocator.deallocate(ptr_dev, n);
}

TEST_CASE("test_locked_pointer","[test_cuda_memory]"){
    using value_type = float;
    using allocator_type = cuda_experimental::locked_allocator<value_type>;
    using difference_type = typename allocator_type::difference_type;
    using cuda_experimental::locked_pointer;

    allocator_type allocator{};
    std::vector<value_type> v{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    auto n = v.size();
    auto ptr_locked = allocator.allocate(n);
    locked_pointer<value_type> ppp{};
    //auto cp = static_cast<locked_pointer<const value_type>>(ppp);
    auto cp = static_cast<locked_pointer<const value_type>>(ptr_locked);
    auto cpp = static_cast<locked_pointer<const char>>(ptr_locked);
    REQUIRE(std::is_same_v<decltype(*ptr_locked),value_type&>);
    //REQUIRE(std::is_same_v<decltype(*static_cast<locked_pointer<const value_type>>(ptr_locked)),const value_type&>);

    // REQUIRE(std::is_same_v<decltype(ptr_locked[std::declval<difference_type>()]),value_type&>);
    // REQUIRE(std::is_same_v<decltype(ptr_to_const(ptr_locked)[std::declval<difference_type>()]),const value_type&>);
    // std::copy(v.begin(), v.end(), ptr_locked);
    // REQUIRE(std::equal(ptr_locked, ptr_locked+n, v.begin()));
    // REQUIRE(std::equal(ptr_to_const(ptr_locked), ptr_to_const(ptr_locked+n), v.begin()));

    // auto transformed = allocator.allocate(n);
    // auto transformator = [](const auto& v){return v+1;};
    // std::vector<value_type> v_expected{2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21};
    // std::transform(ptr_locked,ptr_locked+n,transformed,transformator);
    // REQUIRE(std::equal(v_expected.begin(),v_expected.end(), transformed));
    // allocator.deallocate(transformed,n);

    allocator.deallocate(ptr_locked, n);
}
