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

TEST_CASE("test_device_storage_n_value_constructor","[test_device_storage]"){
    using value_type = float;
    using storage_type = cuda_experimental::device_storage<value_type>;

    auto storage_size = 100;
    auto cuda_storage = storage_type(storage_size, value_type{1.0f});
    REQUIRE(cuda_storage.size() == storage_size);
}

TEST_CASE("test_device_storage_host_range_constructor","[test_device_storage]"){
    using value_type = float;
    using storage_type = cuda_experimental::device_storage<value_type>;

    auto host_data = std::vector<value_type>{1,2,3,4,5,6,7,8,9,10};
    auto cuda_storage = storage_type(host_data.begin(), host_data.end());
    REQUIRE(cuda_storage.size() == host_data.size());
}

