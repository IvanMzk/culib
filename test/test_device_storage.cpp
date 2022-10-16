#include "catch.hpp"
#include "device_storage.hpp"


TEST_CASE("test_device_storage","[test_device_storage]"){
    using value_type = float;
    using storage_type = cuda_experimental::device_storage<value_type>;

    auto cuda_storage = storage_type(100);
    REQUIRE(cuda_storage.size() == 100);
    REQUIRE(cuda_storage.begin());
}