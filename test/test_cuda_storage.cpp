#include <vector>
#include <list>
#include <numeric>
#include "catch.hpp"
#include "cuda_storage.hpp"

TEST_CASE("test_is_iterator","[test_tensor]"){
    using culib::detail::is_iterator;
    using culib::device_pointer;
    REQUIRE(!is_iterator<int>);
    REQUIRE(!is_iterator<std::vector<int>>);
    REQUIRE(is_iterator<device_pointer<float>>);
    REQUIRE(is_iterator<std::vector<int>::iterator>);
    REQUIRE(is_iterator<const float*>);
    REQUIRE(is_iterator<float*>);
}

TEMPLATE_TEST_CASE("test_cuda_storage_default_constructor","[test_cuda_storage]",
    (culib::cuda_storage<float, culib::device_allocator<float>>)
)
{
    using culib::distance;
    using storage_type = TestType;
    auto cuda_storage = storage_type();
    REQUIRE(cuda_storage.size() == 0);
    REQUIRE(distance(cuda_storage.begin(), cuda_storage.end()) == 0);
    REQUIRE(cuda_storage.empty());
}

TEMPLATE_TEST_CASE("test_cuda_storage_n_constructor","[test_cuda_storage]",
    (culib::cuda_storage<float, culib::device_allocator<float>>)
)
{
    using culib::distance;
    using storage_type = TestType;

    SECTION("zero_size"){
        auto cuda_storage = storage_type(0);
        REQUIRE(cuda_storage.size() == 0);
        REQUIRE(distance(cuda_storage.begin(), cuda_storage.end()) == 0);
        REQUIRE(cuda_storage.empty());
    }
    SECTION("not_zero_size"){
        auto storage_size = 100;
        auto cuda_storage = storage_type(storage_size);
        REQUIRE(cuda_storage.size() == storage_size);
        REQUIRE(distance(cuda_storage.begin(), cuda_storage.end()) == storage_size);
        REQUIRE(!cuda_storage.empty());
    }
}

TEMPLATE_TEST_CASE("test_cuda_storage_n_value_constructor","[test_cuda_storage]",
    (culib::cuda_storage<float, culib::device_allocator<float>>)
)
{
    using storage_type = TestType;
    using value_type = typename storage_type::value_type;
    using culib::distance;
    value_type v{11.0f};
    SECTION("non_zero_size"){
        auto n = 100;
        auto cuda_storage = storage_type(n, v);
        REQUIRE(cuda_storage.size() == n);
        REQUIRE(distance(cuda_storage.begin(), cuda_storage.end()) == n);
        REQUIRE(!cuda_storage.empty());
        std::vector<value_type> expected(n, v);
        REQUIRE(std::equal(cuda_storage.begin(), cuda_storage.end(), expected.begin()));
    }
    SECTION("zero_size"){
        auto n = 0;
        auto cuda_storage = storage_type(n, v);
        REQUIRE(cuda_storage.size() == n);
        REQUIRE(distance(cuda_storage.begin(), cuda_storage.end()) == n);
        REQUIRE(cuda_storage.empty());
    }
}

TEMPLATE_TEST_CASE("test_cuda_storage_pointers_range_constructor","[test_cuda_storage]",
    (culib::cuda_storage<float, culib::device_allocator<float>>)
)
{
    using storage_type = TestType;
    using value_type = typename storage_type::value_type;

    SECTION("host_pointers_range"){
        const auto n = 1024;
        std::vector<value_type> host_data(n);
        std::iota(host_data.begin(),host_data.end(),value_type{0});
        SECTION("not_empty_range"){
            auto cuda_storage = storage_type(host_data.data(), host_data.data()+n);
            REQUIRE(cuda_storage.size() == n);
            REQUIRE(!cuda_storage.empty());
            REQUIRE(std::equal(cuda_storage.begin(), cuda_storage.end(), host_data.begin()));
        }
        SECTION("empty_range"){
            auto cuda_storage = storage_type(host_data.data(), host_data.data());
            REQUIRE(cuda_storage.size() == 0);
            REQUIRE(cuda_storage.empty());
        }
    }

    SECTION("cuda_pointers_range"){
        auto expected = storage_type{1,2,3,4,5,6,7,8,9,10};
        SECTION("not_empty_range"){
            auto result = storage_type(expected.begin(),expected.end());
            REQUIRE(result.size() == expected.size());
            REQUIRE(!result.empty());
            REQUIRE(expected.data() != result.data());
            REQUIRE(std::equal(expected.begin(), expected.end(), result.begin()));
        }
        SECTION("empty_range"){
            auto result = storage_type(expected.begin(),expected.begin());
            REQUIRE(result.size() == 0);
            REQUIRE(result.empty());
        }
    }

    SECTION("cuda_peer_pointers_range"){
        using culib::cuda_set_device;
        using culib::cuda_get_device_count;
        if (cuda_get_device_count() > 1){
            constexpr int expected_device_id = 1;
            constexpr int result_device_id = 0;
            cuda_set_device(expected_device_id);
            auto expected = storage_type{1,2,3,4,5,6,7,8,9,10};
            REQUIRE(expected.begin().device() == expected_device_id);
            REQUIRE(expected.end().device() == expected_device_id);
            SECTION("not_empty_range"){
                cuda_set_device(result_device_id);
                auto result = storage_type(expected.begin(),expected.end());
                REQUIRE(result.begin().device() == result_device_id);
                REQUIRE(result.end().device() == result_device_id);
                REQUIRE(result.size() == expected.size());
                REQUIRE(!result.empty());
                REQUIRE(std::equal(expected.begin(), expected.end(), result.begin()));
            }
            SECTION("empty_range"){
                cuda_set_device(result_device_id);
                auto result = storage_type(expected.begin(),expected.begin());
                REQUIRE(result.size() == 0);
                REQUIRE(result.empty());
            }
        }
    }
}

TEMPLATE_TEST_CASE("test_cuda_storage_std_iterators_range_constructor","[test_cuda_storage]",
    std::vector<float>,
    std::list<float>
)
{
    using container_type = TestType;
    using value_type = typename container_type::value_type;
    using storage_type = culib::cuda_storage<value_type, culib::device_allocator<value_type>>;
    using size_type = typename storage_type::size_type;

    const auto n = 1024;
    container_type expected(n);
    std::iota(expected.begin(),expected.end(),value_type{0});
    SECTION("not_empty_range"){
        storage_type cuda_storage(expected.begin(), expected.end());
        REQUIRE(cuda_storage.size() == static_cast<size_type>(expected.size()));
        REQUIRE(!cuda_storage.empty());
        REQUIRE(std::equal(cuda_storage.begin(), cuda_storage.end(), expected.begin()));
    }
    SECTION("empty_range"){
        storage_type cuda_storage(expected.begin(), expected.begin());
        REQUIRE(cuda_storage.size() == 0);
        REQUIRE(cuda_storage.empty());
    }
}

TEMPLATE_TEST_CASE("test_cuda_storage_init_list_constructor","[test_cuda_storage]",
    (culib::cuda_storage<float, culib::device_allocator<float>>)
)
{
    using storage_type = TestType;
    using value_type = typename storage_type::value_type;
    auto cuda_storage = storage_type({1,2,3,4,5,6,7,8,9,10});
    REQUIRE(cuda_storage.size() == 10);
    REQUIRE(!cuda_storage.empty());
    REQUIRE(std::equal(cuda_storage.begin(), cuda_storage.end(), std::initializer_list<value_type>{1,2,3,4,5,6,7,8,9,10}.begin()));
}

TEMPLATE_TEST_CASE("test_cuda_storage_free","[test_cuda_storage]",
    (culib::cuda_storage<float, culib::device_allocator<float>>)
)
{
    using storage_type = TestType;
    auto cuda_storage = storage_type({1,2,3,4,5,6,7,8,9,10});
    cuda_storage.free();
    REQUIRE(cuda_storage.size() == 0);
    REQUIRE(cuda_storage.empty());
}

TEMPLATE_TEST_CASE("test_cuda_storage_copy_constructor","[test_cuda_storage]",
    (culib::cuda_storage<float, culib::device_allocator<float>>)
)
{
    using storage_type = TestType;
    using value_type = typename storage_type::value_type;

    auto storage_size = 100;
    auto cuda_storage = storage_type(storage_size, value_type{1.0f});
    auto copy = cuda_storage.clone();
    REQUIRE(copy.size() == storage_size);
    REQUIRE(cuda_storage.size() == storage_size);
    REQUIRE(!copy.empty());
    REQUIRE(!cuda_storage.empty());
    REQUIRE(copy.data() != cuda_storage.data());
    REQUIRE(std::equal(cuda_storage.begin(), cuda_storage.end(), copy.begin()));
}

TEMPLATE_TEST_CASE("test_cuda_storage_copy_assignment","[test_cuda_storage]",
    (culib::cuda_storage<float, culib::device_allocator<float>>)
)
{
    using storage_type = TestType;
    using size_type = typename storage_type::size_type;
    using value_type = typename storage_type::value_type;
    using allocator_type = typename storage_type::allocator_type;
    REQUIRE(!typename std::allocator_traits<allocator_type>::propagate_on_container_copy_assignment());
    REQUIRE(typename std::allocator_traits<allocator_type>::is_always_equal());
    static constexpr size_type n{100};
    auto cuda_storage = storage_type(n,7);

    SECTION("not_self_assignment_reallocation"){
        auto copy_assigned_size = GENERATE(n-1, n+1);
        auto copy_assigned = storage_type(copy_assigned_size,0);
        auto initial_copy_assigned_data = copy_assigned.data();
        copy_assigned = cuda_storage;
        REQUIRE(copy_assigned.data() != initial_copy_assigned_data);
        REQUIRE(copy_assigned.data() != cuda_storage.data());
        REQUIRE(cuda_storage.size() == n);
        REQUIRE(!cuda_storage.empty());
        REQUIRE(copy_assigned.size() == n);
        REQUIRE(!copy_assigned.empty());
        REQUIRE(std::equal(copy_assigned.begin(), copy_assigned.end(), cuda_storage.begin()));
    }
    SECTION("not_self_assignment_no_reallocation"){
        auto copy_assigned_size = GENERATE(n+0);
        auto copy_assigned = storage_type(copy_assigned_size,0);
        auto initial_copy_assigned_data = copy_assigned.data();
        copy_assigned = cuda_storage;
        REQUIRE(copy_assigned.data() == initial_copy_assigned_data);
        REQUIRE(copy_assigned.data() != cuda_storage.data());
        REQUIRE(cuda_storage.size() == n);
        REQUIRE(!cuda_storage.empty());
        REQUIRE(copy_assigned.size() == n);
        REQUIRE(!copy_assigned.empty());
        REQUIRE(std::equal(copy_assigned.begin(), copy_assigned.end(), cuda_storage.begin()));
    }
    SECTION("self_assignment"){
        auto initial_cuda_storage_data = cuda_storage.data();
        auto& copy_assigned = cuda_storage;
        copy_assigned = cuda_storage;
        REQUIRE(&copy_assigned == &cuda_storage);
        REQUIRE(copy_assigned.size() == n);
        REQUIRE(copy_assigned.data() == initial_cuda_storage_data);
    }
}

TEMPLATE_TEST_CASE("test_cuda_storage_move_constructor","[test_cuda_storage]",
    (culib::cuda_storage<float, culib::device_allocator<float>>)
)
{
    using storage_type = TestType;
    using value_type = typename storage_type::value_type;

    auto storage_size = 100;
    auto cuda_storage = storage_type(storage_size, value_type{1.0f});
    auto data = cuda_storage.data();
    auto copy_moved = std::move(cuda_storage);
    REQUIRE(!copy_moved.empty());
    REQUIRE(copy_moved.size() == storage_size);
    REQUIRE(copy_moved.data() == data);
    REQUIRE(cuda_storage.empty());
    REQUIRE(cuda_storage.size() == 0);
}

TEMPLATE_TEST_CASE("test_cuda_storage_move_assignment","[test_cuda_storage]",
    (culib::cuda_storage<float, culib::device_allocator<float>>)
)
{
    using storage_type = TestType;
    using size_type = typename storage_type::size_type;
    using value_type = typename storage_type::value_type;
    using allocator_type = typename storage_type::allocator_type;
    REQUIRE(!typename std::allocator_traits<allocator_type>::propagate_on_container_move_assignment());
    REQUIRE(typename std::allocator_traits<allocator_type>::is_always_equal());
    size_type n{10};
    value_type v{3};
    auto cuda_storage = storage_type(n,v);

    auto move_assigned = storage_type(n+10,0);
    move_assigned = std::move(cuda_storage);
    REQUIRE(move_assigned.size() == n);
    REQUIRE(!move_assigned.empty());
    REQUIRE(std::equal(move_assigned.begin(), move_assigned.end(), std::vector<value_type>(n,v).begin()));
    REQUIRE(cuda_storage.size() == 0);
    REQUIRE(cuda_storage.empty());
}


