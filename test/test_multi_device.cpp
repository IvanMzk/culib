#include <iostream>
#include "catch.hpp"
#include "cuda_helpers.hpp"

TEST_CASE("test_multi_gpu","[test_cuda_aware_storage]"){
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