#include "catch.hpp"
#include "thread_pool.hpp"

namespace test_thread_pool{

    auto add(int a,int b){
        return a+b;
    }

    auto mul(int a,int b){
        return a*b;
    }

}   //end of namespace test_thread_pool


TEST_CASE("test_thread_pool","[test_thread_pool]"){

    using test_thread_pool::add;
    using test_thread_pool::mul;
    using thread_pool_type = cuda_experimental::thread_pool<2,10,int(int,int)>;

    thread_pool_type tpool{};
    REQUIRE(tpool.size() == 0);
    tpool.push(add, 1,1);
    REQUIRE(tpool.size() == 1);
    tpool.push(mul, 2,2);
    REQUIRE(tpool.size() == 2);

}