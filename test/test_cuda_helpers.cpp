#include "catch.hpp"
#include "cuda_helpers.hpp"



TEST_CASE("test_error_check","[test_cuda_helpers]"){
    using culib::cuda_assert;
    using culib::cuda_exception;
    REQUIRE_NOTHROW(cuda_assert(cudaError::cudaSuccess,"",0));
    REQUIRE_THROWS_AS(cuda_assert(cudaError::cudaErrorAssert,"",0),cuda_exception);
}