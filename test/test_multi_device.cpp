#include <iostream>
#include "catch.hpp"
#include "cuda_helpers.hpp"

TEST_CASE("test_multi_gpu","[test_multi_gpu]"){
    using cuda_experimental::cuda_get_device_count;
    auto n = cuda_get_device_count();
    if (n == 0){
        std::cout<<std::endl<<"NO DEVICE DETECTED, DEVICE TESTS WILL THROW"<<std::endl;
    }else if(n == 1){
        std::cout<<std::endl<<"1 DEVICE DETECTED, MULTI DEVICE TESTS WILL NOT RUN"<<std::endl;
    }else{
        std::cout<<std::endl<<n<<" DEVICES DETECTED, MULTI DEVICE TESTS WILL RUN"<<std::endl;
    }
}

TEST_CASE("test_peer_access","[test_multi_gpu]"){
    using cuda_experimental::cuda_get_device_count;
    using cuda_experimental::cuda_device_can_access_peer;
    auto n = cuda_get_device_count();
    for (int i{0}; i!=n; ++i){
        for (int j{0}; j!=n; ++j){
            auto is_enabled_from_i_to_j = cuda_device_can_access_peer(i,j);
            std::cout<<std::endl<<"DEVICE ACCESS FROM "<<i<<" DEVICE ACCESS TO "<<j<<" "<<is_enabled_from_i_to_j;
        }
    }
}

TEST_CASE("test_properties","[test_multi_gpu]"){
    using cuda_experimental::cuda_get_device_count;
    using cuda_experimental::cuda_get_device_properties;
    auto n = cuda_get_device_count();
    for (int i{0}; i!=n; ++i){
        auto prop = cuda_get_device_properties(i);
        std::cout<<std::endl<<"DEVICE "<<i<<" unifiedAddressing "<<prop.unifiedAddressing;
    }
}