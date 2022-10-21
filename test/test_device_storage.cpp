#include <vector>
#include <list>
#include "catch.hpp"
#include "device_storage.hpp"


TEST_CASE("test_device_storage_default_constructor","[test_device_storage]"){
    using value_type = float;
    using cuda_experimental::distance;
    using storage_type = cuda_experimental::device_storage<value_type>;
    auto cuda_storage = storage_type();
    REQUIRE(cuda_storage.size() == 0);
    REQUIRE(distance(cuda_storage.device_begin(), cuda_storage.device_end()) == 0);
    REQUIRE(cuda_storage.empty());
}

TEST_CASE("test_device_storage_n_constructor","[test_device_storage]"){
    using value_type = float;
    using cuda_experimental::distance;
    using storage_type = cuda_experimental::device_storage<value_type>;

    SECTION("zero_size"){
        auto cuda_storage = storage_type(0);
        REQUIRE(cuda_storage.size() == 0);
        REQUIRE(distance(cuda_storage.device_begin(), cuda_storage.device_end()) == 0);
        REQUIRE(cuda_storage.empty());
    }
    SECTION("not_zero_size"){
        auto storage_size = 100;
        auto cuda_storage = storage_type(storage_size);
        REQUIRE(cuda_storage.size() == storage_size);
        REQUIRE(distance(cuda_storage.device_begin(), cuda_storage.device_end()) == storage_size);
        REQUIRE(!cuda_storage.empty());
    }
}

TEST_CASE("test_device_storage_n_value_constructor","[test_device_storage]"){
    using value_type = float;
    using cuda_experimental::distance;
    using storage_type = cuda_experimental::device_storage<value_type>;
    SECTION("non_zero_size"){
        auto storage_size = 100;
        auto cuda_storage = storage_type(storage_size, value_type{1.0f});
        REQUIRE(cuda_storage.size() == storage_size);
        REQUIRE(distance(cuda_storage.device_begin(), cuda_storage.device_end()) == storage_size);
        REQUIRE(!cuda_storage.empty());
    }
    SECTION("zero_size"){
        auto storage_size = 0;
        auto cuda_storage = storage_type(storage_size, value_type{1.0f});
        REQUIRE(cuda_storage.size() == storage_size);
        REQUIRE(distance(cuda_storage.device_begin(), cuda_storage.device_end()) == storage_size);
        REQUIRE(cuda_storage.empty());
    }
}

TEST_CASE("test_device_storage_copy_constructor","[test_device_storage]"){
    using value_type = float;
    using storage_type = cuda_experimental::device_storage<value_type>;

    auto storage_size = 100;
    auto cuda_storage = storage_type(storage_size, value_type{1.0f});
    auto copy = cuda_storage.clone();
    REQUIRE(copy.size() == storage_size);
    REQUIRE(cuda_storage.size() == storage_size);
    REQUIRE(!copy.empty());
    REQUIRE(!cuda_storage.empty());
    REQUIRE(copy.data() != cuda_storage.data());
}

TEST_CASE("test_device_storage_move_constructor","[test_device_storage]"){
    using value_type = float;
    using storage_type = cuda_experimental::device_storage<value_type>;

    auto storage_size = 100;
    auto cuda_storage = storage_type(storage_size, value_type{1.0f});
    auto data = cuda_storage.data();
    auto move = std::move(cuda_storage);
    REQUIRE(!move.empty());
    REQUIRE(move.size() == storage_size);
    REQUIRE(move.data() == data);
    REQUIRE(cuda_storage.empty());
    REQUIRE(cuda_storage.size() == 0);
}

TEST_CASE("test_device_storage_host_range_constructor","[test_device_storage]"){
    using value_type = float;
    using storage_type = cuda_experimental::device_storage<value_type>;

    SECTION("pointer_iter"){
        constexpr std::size_t n{100};
        value_type host_data[n]{};
        SECTION("not_empty_range"){
            auto cuda_storage = storage_type(host_data, host_data+n);
            REQUIRE(cuda_storage.size() == n);
            REQUIRE(!cuda_storage.empty());
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
        }
        SECTION("empty_range"){
            auto cuda_storage = storage_type(host_data.begin(), host_data.begin());
            REQUIRE(cuda_storage.size() == 0);
            REQUIRE(cuda_storage.empty());
        }
    }
}

TEST_CASE("test_device_storage_init_list_constructor","[test_device_storage]"){
    using value_type = float;
    using storage_type = cuda_experimental::device_storage<value_type>;

    auto cuda_storage = storage_type({1,2,3,4,5,6,7,8,9,10});
    REQUIRE(cuda_storage.size() == 10);
    REQUIRE(!cuda_storage.empty());
}

TEST_CASE("test_device_storage_device_range_constructor","[test_device_storage]"){
    using value_type = float;
    using storage_type = cuda_experimental::device_storage<value_type>;

    auto cuda_storage = storage_type({1,2,3,4,5,6,7,8,9,10});
    SECTION("not_empty_range"){
        auto storage_from_device_range = storage_type(cuda_storage.device_begin()+1,cuda_storage.device_end()-1);
        REQUIRE(storage_from_device_range.size() == 8);
        REQUIRE(!storage_from_device_range.empty());
    }
    SECTION("empty_range"){
        auto storage_from_device_range = storage_type(cuda_storage.device_begin(),cuda_storage.device_begin());
        REQUIRE(storage_from_device_range.size() == 0);
        REQUIRE(storage_from_device_range.empty());
    }
}

TEST_CASE("test_device_storage_free","[test_device_storage]"){
    using value_type = float;
    using storage_type = cuda_experimental::device_storage<value_type>;

    auto cuda_storage = storage_type({1,2,3,4,5,6,7,8,9,10});
    cuda_storage.free();
    REQUIRE(cuda_storage.size() == 0);
    REQUIRE(cuda_storage.empty());
}

