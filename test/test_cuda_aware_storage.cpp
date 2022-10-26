#include <vector>
#include <list>
#include "catch.hpp"
#include "cuda_aware_storage.hpp"

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

TEMPLATE_TEST_CASE("test_cuda_aware_storage_default_constructor","[test_cuda_aware_storage]",
    (cuda_experimental::cuda_aware_storage<float, cuda_experimental::device_allocator<float>>)
)
{
    using value_type = float;
    using cuda_experimental::distance;
    using storage_type = TestType;
    auto cuda_storage = storage_type();
    REQUIRE(cuda_storage.size() == 0);
    REQUIRE(distance(cuda_storage.begin(), cuda_storage.end()) == 0);
    REQUIRE(cuda_storage.empty());
}

TEMPLATE_TEST_CASE("test_cuda_aware_storage_n_constructor","[test_cuda_aware_storage]",
    (cuda_experimental::cuda_aware_storage<float, cuda_experimental::device_allocator<float>>)
)
{
    using value_type = float;
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

TEMPLATE_TEST_CASE("test_cuda_aware_storage_n_value_constructor","[test_cuda_aware_storage]",
    (cuda_experimental::cuda_aware_storage<float, cuda_experimental::device_allocator<float>>)
)
{
    using value_type = float;
    using cuda_experimental::distance;
    using storage_type = TestType;
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

TEMPLATE_TEST_CASE("test_cuda_aware_storage_host_range_constructor","[test_cuda_aware_storage]",
    (cuda_experimental::cuda_aware_storage<float, cuda_experimental::device_allocator<float>>)
)
{
    using value_type = float;
    using storage_type = TestType;

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

TEMPLATE_TEST_CASE("test_cuda_aware_storage_init_list_constructor","[test_cuda_aware_storage]",
    (cuda_experimental::cuda_aware_storage<float, cuda_experimental::device_allocator<float>>)
)
{
    using value_type = float;
    using storage_type = TestType;
    auto cuda_storage = storage_type({1,2,3,4,5,6,7,8,9,10});
    REQUIRE(cuda_storage.size() == 10);
    REQUIRE(!cuda_storage.empty());
    REQUIRE(std::equal(cuda_storage.begin(), cuda_storage.end(), std::initializer_list<float>{1,2,3,4,5,6,7,8,9,10}.begin()));
}

TEMPLATE_TEST_CASE("test_cuda_aware_storage_free","[test_cuda_aware_storage]",
    (cuda_experimental::cuda_aware_storage<float, cuda_experimental::device_allocator<float>>)
)
{
    using value_type = float;
    using storage_type = TestType;

    auto cuda_storage = storage_type({1,2,3,4,5,6,7,8,9,10});
    cuda_storage.free();
    REQUIRE(cuda_storage.size() == 0);
    REQUIRE(cuda_storage.empty());
}

// TEST_CASE("test_cuda_aware_storage_device_range_constructor","[test_cuda_aware_storage]"){
//     using value_type = float;
//     using storage_type = cuda_experimental::cuda_aware_storage<value_type>;

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

// TEST_CASE("test_cuda_aware_storage_copy_constructor","[test_cuda_aware_storage]"){
//     using value_type = float;
//     using storage_type = cuda_experimental::cuda_aware_storage<value_type>;

//     auto storage_size = 100;
//     auto cuda_storage = storage_type(storage_size, value_type{1.0f});
//     auto copy = cuda_storage.clone();
//     REQUIRE(copy.size() == storage_size);
//     REQUIRE(cuda_storage.size() == storage_size);
//     REQUIRE(!copy.empty());
//     REQUIRE(!cuda_storage.empty());
//     REQUIRE(copy.data() != cuda_storage.data());
// }

// TEST_CASE("test_cuda_aware_storage_move_constructor","[test_cuda_aware_storage]"){
//     using value_type = float;
//     using storage_type = cuda_experimental::cuda_aware_storage<value_type>;

//     auto storage_size = 100;
//     auto cuda_storage = storage_type(storage_size, value_type{1.0f});
//     auto data = cuda_storage.data();
//     auto move = std::move(cuda_storage);
//     REQUIRE(!move.empty());
//     REQUIRE(move.size() == storage_size);
//     REQUIRE(move.data() == data);
//     REQUIRE(cuda_storage.empty());
//     REQUIRE(cuda_storage.size() == 0);
// }

// TEST_CASE("test_cuda_aware_storage_copy_assignment","[test_cuda_aware_storage]"){
//     using value_type = float;
//     using storage_type = cuda_experimental::cuda_aware_storage<value_type>;

//     auto n = std::size_t{10};
//     auto cuda_storage = storage_type(n,1);

//     SECTION("not_equal_size_realloction"){
//         auto storage_copy = storage_type(n+10,0);
//         REQUIRE(storage_copy.size() != n);
//         storage_copy = cuda_storage;
//         REQUIRE(cuda_storage.size() == n);
//         REQUIRE(!cuda_storage.empty());
//         REQUIRE(storage_copy.size() == n);
//         REQUIRE(!storage_copy.empty());
//         REQUIRE(storage_copy.data() != cuda_storage.data());
//     }
//     SECTION("equal_size_no_realloction"){
//         auto storage_copy = storage_type(n,0);
//         REQUIRE(storage_copy.size() == n);
//         storage_copy = cuda_storage;
//         REQUIRE(cuda_storage.size() == n);
//         REQUIRE(!cuda_storage.empty());
//         REQUIRE(storage_copy.size() == n);
//         REQUIRE(!storage_copy.empty());
//         REQUIRE(storage_copy.data() != cuda_storage.data());
//     }
// }

// TEST_CASE("test_cuda_aware_storage_move_assignment","[test_cuda_aware_storage]"){
//     using value_type = float;
//     using storage_type = cuda_experimental::cuda_aware_storage<value_type>;

//     auto n = std::size_t{10};
//     auto cuda_storage = storage_type(n,1);

//     auto storage_copy = storage_type(n+10,0);
//     storage_copy = std::move(cuda_storage);
//     REQUIRE(storage_copy.size() == n);
//     REQUIRE(!storage_copy.empty());
//     REQUIRE(cuda_storage.size() == 0);
//     REQUIRE(cuda_storage.empty());
// }