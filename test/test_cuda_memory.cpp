#include "catch.hpp"
#include "cuda_memory.hpp"


TEST_CASE("test_cuda_pointer","[test_cuda_memory]"){
    using value_type = float;
    using difference_type = std::ptrdiff_t;
    using cuda_pointer_type = cuda_experimental::cuda_pointer<value_type,difference_type>;

    auto v = float{};
    auto p = cuda_pointer_type{};
    REQUIRE(p.get() == nullptr);

    SECTION("from_pointer_construction"){
        auto p = cuda_pointer_type{&v};
        REQUIRE(p.get() == &v);
        auto p1 = cuda_pointer_type{nullptr};
        REQUIRE(p1.get() == nullptr);
    }
    SECTION("copy_construction"){
        auto p = cuda_pointer_type{&v};
        auto p1 = p;
        REQUIRE(p1.get() == &v);
    }
    SECTION("assignment"){
        auto p = cuda_pointer_type{};
        auto p1 = cuda_pointer_type{&v};
        p = p1;
        REQUIRE(p.get() == &v);
        REQUIRE(p1.get() == &v);
        p = nullptr;
        REQUIRE(p.get() == nullptr);
    }
    SECTION("equality"){
        auto p = cuda_pointer_type{};
        auto p1 = cuda_pointer_type{&v};
        auto p2 = cuda_pointer_type{&v};
        REQUIRE(p1 == p2);
        REQUIRE(p1 != p);
        REQUIRE(p2 != p);
    }
    SECTION("add_offset"){
        value_type a[10];
        auto p = cuda_pointer_type{a};
        REQUIRE(p+5 == cuda_pointer_type{a+5});
        REQUIRE(5+p == cuda_pointer_type{a+5});
    }
    SECTION("subtract_two_pointers"){
        value_type a[10];
        auto begin = cuda_pointer_type{a};
        auto end = cuda_pointer_type{a+10};
        REQUIRE(end-begin == 10);
        REQUIRE(end-end == 0);
        REQUIRE(distance(begin,end) == 10);
        REQUIRE(distance(begin,begin) == 0);
        REQUIRE(distance(begin,begin+3) == 3);
    }
    SECTION("distance"){
        value_type a[10];
        auto begin = cuda_pointer_type{a};
        auto end = cuda_pointer_type{a+10};
        REQUIRE(distance(begin,end) == 10);
        REQUIRE(distance(begin,begin) == 0);
        REQUIRE(distance(begin,begin+3) == 3);
        REQUIRE(distance(begin+2,end-3) == 5);
    }
}