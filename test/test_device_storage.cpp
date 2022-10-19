#include <vector>
#include "catch.hpp"
#include "device_storage.hpp"


TEST_CASE("test_device_storage","[test_device_storage]"){
    using value_type = float;
    using storage_type = cuda_experimental::device_storage<value_type>;

    auto storage_size = 100;
    auto cuda_storage = storage_type(storage_size);
    REQUIRE(cuda_storage.size() == storage_size);
    REQUIRE(cuda_storage.device_end() == cuda_storage.device_begin()+cuda_storage.size());

    SECTION("test_free"){
        cuda_storage.free();
        REQUIRE(cuda_storage.size() == 0);
        REQUIRE(cuda_storage.device_begin() == cuda_storage.device_end());
    }
    SECTION("test_clone"){
        auto copy = cuda_storage.clone();
        REQUIRE(copy.size() == cuda_storage.size());
        REQUIRE(copy.device_begin() != cuda_storage.device_begin());
    }
    SECTION("test_move"){
        auto move = std::move(cuda_storage);
        REQUIRE(move.size() == storage_size);
        REQUIRE(move.device_end() == move.device_begin()+move.size());
        REQUIRE(cuda_storage.size() == 0);
        REQUIRE(cuda_storage.device_begin() == cuda_storage.device_end());
    }
}

TEST_CASE("test_device_storage_default_constructor","[test_device_storage]"){
    using value_type = float;
    using cuda_experimental::distance;
    using storage_type = cuda_experimental::device_storage<value_type>;
    auto cuda_storage = storage_type();
    REQUIRE(cuda_storage.size() == 0);
    REQUIRE(distance(cuda_storage.device_begin(), cuda_storage.device_end()) == 0);
    REQUIRE(cuda_storage.empty());
}

TEST_CASE("test_device_storage_n_value_constructor","[test_device_storage]"){
    using value_type = float;
    using cuda_experimental::distance;
    using storage_type = cuda_experimental::device_storage<value_type>;

    auto storage_size = 100;
    auto cuda_storage = storage_type(storage_size, value_type{1.0f});
    REQUIRE(cuda_storage.size() == storage_size);
    REQUIRE(distance(cuda_storage.device_begin(), cuda_storage.device_end()) == storage_size);
    REQUIRE(!cuda_storage.empty());
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

    auto host_data = std::vector<value_type>{1,2,3,4,5,6,7,8,9,10};
    auto cuda_storage = storage_type(host_data.begin(), host_data.end());
    REQUIRE(cuda_storage.size() == host_data.size());
    REQUIRE(!cuda_storage.empty());
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
    auto storage_from_device_range = storage_type(cuda_storage.device_begin()+1,cuda_storage.device_end()-1);
    REQUIRE(storage_from_device_range.size() == 8);
    REQUIRE(!storage_from_device_range.empty());
}

TEST_CASE("test_device_storage_free","[test_device_storage]"){
    using value_type = float;
    using storage_type = cuda_experimental::device_storage<value_type>;

    auto cuda_storage = storage_type({1,2,3,4,5,6,7,8,9,10});
    cuda_storage.free();
    REQUIRE(cuda_storage.size() == 0);
    REQUIRE(cuda_storage.empty());
}

