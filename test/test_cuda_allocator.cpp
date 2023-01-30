#include <array>
#include "catch.hpp"
#include "cuda_memory.hpp"

namespace test_cuda_allocator{

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

}   //end of namespace test_cuda_allocator

TEMPLATE_TEST_CASE("test_device_allocator","[test_cuda_allocator]",
    float,
    (std::array<int,4>),
    (test_cuda_allocator::test_array<test_cuda_allocator::test_lin_space<double>,10>)
)
{
    using value_type = TestType;
    using allocator_type = cuda_experimental::device_allocator<value_type>;
    using cuda_experimental::cuda_get_device;

    std::size_t n{100};
    allocator_type allocator{};
    auto dev = cuda_get_device();
    auto ptr = allocator.allocate(n);
    REQUIRE(ptr.get() != nullptr);
    REQUIRE(ptr.device() == dev);
    allocator.deallocate(ptr,n);
}

TEMPLATE_TEST_CASE("test_locked_allocator","[test_cuda_allocator]",
    float,
    (std::array<int,4>),
    (test_cuda_allocator::test_array<test_cuda_allocator::test_lin_space<double>,10>)
)
{
    using value_type = TestType;
    using allocator_type = cuda_experimental::locked_allocator<value_type>;

    std::size_t n{100};
    allocator_type allocator{};
    auto ptr = allocator.allocate(n);
    REQUIRE(ptr.get() != nullptr);
    allocator.deallocate(ptr,n);
}
