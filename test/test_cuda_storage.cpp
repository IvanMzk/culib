#include <vector>
#include <list>
#include "catch.hpp"
#include "cuda_storage.hpp"

TEST_CASE("test_is_iterator","[test_tensor]"){
    using cuda_experimental::detail::is_iterator;
    using cuda_experimental::device_pointer;
    REQUIRE(!is_iterator<int>);
    REQUIRE(!is_iterator<std::vector<int>>);
    REQUIRE(is_iterator<device_pointer<float>>);
    REQUIRE(is_iterator<std::vector<int>::iterator>);
    REQUIRE(is_iterator<const float*>);
    REQUIRE(is_iterator<float*>);
}

TEMPLATE_TEST_CASE("test_cuda_storage_default_constructor","[test_cuda_storage]",
    (cuda_experimental::cuda_storage<float, cuda_experimental::device_allocator<float>>)
)
{
    using cuda_experimental::distance;
    using storage_type = TestType;
    auto cuda_storage = storage_type();
    REQUIRE(cuda_storage.size() == 0);
    REQUIRE(distance(cuda_storage.begin(), cuda_storage.end()) == 0);
    REQUIRE(cuda_storage.empty());
}

TEMPLATE_TEST_CASE("test_cuda_storage_n_constructor","[test_cuda_storage]",
    (cuda_experimental::cuda_storage<float, cuda_experimental::device_allocator<float>>)
)
{
    using cuda_experimental::distance;
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
    (cuda_experimental::cuda_storage<float, cuda_experimental::device_allocator<float>>)
)
{
    using storage_type = TestType;
    using value_type = typename storage_type::value_type;
    using cuda_experimental::distance;
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

TEMPLATE_TEST_CASE("test_cuda_storage_host_range_constructor","[test_cuda_storage]",
    (cuda_experimental::cuda_storage<float, cuda_experimental::device_allocator<float>>)
)
{
    using storage_type = TestType;
    using value_type = typename storage_type::value_type;

    SECTION("pointers_range"){
        value_type host_data[]{1,2,3,4,5,6,7,8,9,10};
        constexpr std::size_t n{sizeof(host_data)/sizeof(value_type)};
        SECTION("not_empty_range"){
            auto cuda_storage = storage_type(host_data, host_data+n);
            REQUIRE(cuda_storage.size() == n);
            REQUIRE(!cuda_storage.empty());
            REQUIRE(std::equal(cuda_storage.begin(), cuda_storage.end(), host_data));
        }
        SECTION("empty_range"){
            auto cuda_storage = storage_type(host_data, host_data);
            REQUIRE(cuda_storage.size() == 0);
            REQUIRE(cuda_storage.empty());
        }
    }
    SECTION("not_pointer_iter"){
        auto host_data = std::list<value_type>{1,2,3,4,5,6,7,8,9,10};
        SECTION("not_empty_range"){
            auto cuda_storage = storage_type(host_data.begin(), host_data.end());
            REQUIRE(cuda_storage.size() == host_data.size());
            REQUIRE(!cuda_storage.empty());
            REQUIRE(std::equal(cuda_storage.begin(), cuda_storage.end(), host_data.begin()));
        }
        SECTION("empty_range"){
            auto cuda_storage = storage_type(host_data.begin(), host_data.begin());
            REQUIRE(cuda_storage.size() == 0);
            REQUIRE(cuda_storage.empty());
        }
    }
}

TEMPLATE_TEST_CASE("test_cuda_storage_init_list_constructor","[test_cuda_storage]",
    (cuda_experimental::cuda_storage<float, cuda_experimental::device_allocator<float>>)
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
    (cuda_experimental::cuda_storage<float, cuda_experimental::device_allocator<float>>)
)
{
    using storage_type = TestType;
    auto cuda_storage = storage_type({1,2,3,4,5,6,7,8,9,10});
    cuda_storage.free();
    REQUIRE(cuda_storage.size() == 0);
    REQUIRE(cuda_storage.empty());
}

TEMPLATE_TEST_CASE("test_cuda_storage_copy_constructor","[test_cuda_storage]",
    (cuda_experimental::cuda_storage<float, cuda_experimental::device_allocator<float>>)
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
    (cuda_experimental::cuda_storage<float, cuda_experimental::device_allocator<float>>)
)
{
    using storage_type = TestType;
    using value_type = typename storage_type::value_type;
    using allocator_type = typename storage_type::allocator_type;
    REQUIRE(!std::allocator_traits<allocator_type>::propagate_on_container_copy_assignment());
    REQUIRE(std::allocator_traits<allocator_type>::is_always_equal());
    static constexpr std::size_t n{100};
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
    (cuda_experimental::cuda_storage<float, cuda_experimental::device_allocator<float>>)
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
    (cuda_experimental::cuda_storage<float, cuda_experimental::device_allocator<float>>)
)
{
    using storage_type = TestType;
    using value_type = typename storage_type::value_type;
    using allocator_type = typename storage_type::allocator_type;
    REQUIRE(!std::allocator_traits<allocator_type>::propagate_on_container_move_assignment());
    REQUIRE(std::allocator_traits<allocator_type>::is_always_equal());
    std::size_t n{10};
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

// TEST_CASE("test_cuda_storage_device_range_constructor","[test_cuda_storage]"){
//     using value_type = float;
//     using storage_type = cuda_experimental::cuda_storage<value_type>;

//     auto cuda_storage = storage_type({1,2,3,4,5,6,7,8,9,10});
//     SECTION("not_empty_range"){
//         auto storage_from_device_range = storage_type(cuda_storage.begin()+1,cuda_storage.end()-1);
//         REQUIRE(storage_from_device_range.size() == 8);
//         REQUIRE(!storage_from_device_range.empty());
//     }
//     SECTION("empty_range"){
//         auto storage_from_device_range = storage_type(cuda_storage.begin(),cuda_storage.begin());
//         REQUIRE(storage_from_device_range.size() == 0);
//         REQUIRE(storage_from_device_range.empty());
//     }
// }




