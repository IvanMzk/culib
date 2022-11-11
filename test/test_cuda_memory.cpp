#include <iostream>
#include <array>
#include "catch.hpp"
#include "cuda_memory.hpp"

namespace test_cuda_memory{
using cuda_experimental::basic_pointer;
template<typename T>
class test_pointer : public basic_pointer<T, test_pointer>
{
public:
    using typename basic_pointer::pointer;
    using typename basic_pointer::value_type;
    operator test_pointer<const T>()const{return test_pointer<const T>{get()};}
    using basic_pointer::operator=;
    test_pointer(pointer p = nullptr):
        basic_pointer(p)
    {}
};

template<typename T, std::size_t N>
class test_array
{
    T elements_[N]{};
public:
    test_array() = default;
    explicit test_array(const T& v){
        std::fill_n(elements_,N,v);
    }
    constexpr std::size_t size()const{return N;}
    bool operator==(const test_array& other){return std::equal(elements_,elements_+N, other.elements_);}
};

template<typename T>
class test_lin_space
{
    T min_;
    T max_;
    std::size_t points_number_;
public:
    test_lin_space(const T& min__, const T& max__, std::size_t points_number__):
        min_{min__},
        max_{max__},
        points_number_{points_number__}
    {}
};

}   //end of namespace test_cuda_memory

TEMPLATE_TEST_CASE("test_basic_pointer","[test_cuda_memory]",
    test_cuda_memory::test_pointer<float>,
    test_cuda_memory::test_pointer<const float>
)
{
    using value_type = float;
    using pointer_type = TestType;

    REQUIRE(std::is_trivially_copyable_v<pointer_type>);
    value_type v{};
    auto p = pointer_type{};
    REQUIRE(p.get() == nullptr);
    SECTION("from_pointer_construction"){
        auto p = pointer_type{&v};
        REQUIRE(p.get() == &v);
        auto p1 = pointer_type{nullptr};
        REQUIRE(p1.get() == nullptr);
    }
    SECTION("copy_construction"){
        auto p = pointer_type{&v};
        auto p1 = p;
        REQUIRE(p1.get() == p.get());
    }
    SECTION("copy_assignment"){
        auto p = pointer_type{};
        auto p1 = pointer_type{&v};
        p = p1;
        REQUIRE(p.get() == p1.get());
        p1 = nullptr;
        REQUIRE(p1.get() == nullptr);
    }
    SECTION("equality"){
        auto p = pointer_type{};
        auto p1 = pointer_type{&v};
        auto p2 = pointer_type{&v};
        REQUIRE(p == p);
        REQUIRE(p1 == p1);
        REQUIRE(p1 == p2);
        REQUIRE(p1 != p);
    }
    SECTION("add_offset"){
        value_type a[10];
        auto p = pointer_type{a};
        REQUIRE(std::is_same_v<decltype(p+5),pointer_type>);
        REQUIRE(std::is_same_v<decltype(5+p),pointer_type>);
        REQUIRE(p+5 == pointer_type{a+5});
        REQUIRE(5+p == pointer_type{a+5});
    }
    SECTION("subtract_two_pointers"){
        value_type a[10];
        auto begin = pointer_type{a};
        auto end = pointer_type{a+10};
        REQUIRE(end-begin == 10);
        REQUIRE(end-end == 0);
        REQUIRE(distance(begin,end) == 10);
        REQUIRE(distance(begin,begin) == 0);
        REQUIRE(distance(begin,begin+3) == 3);
    }
    SECTION("distance"){
        value_type a[10];
        auto begin = pointer_type{a};
        auto end = pointer_type{a+10};
        REQUIRE(distance(begin,end) == 10);
        REQUIRE(distance(begin,begin) == 0);
        REQUIRE(distance(begin,begin+3) == 3);
        REQUIRE(distance(begin+2,end-3) == 5);
    }
    SECTION("bool_cast_operator"){
        pointer_type p{};
        REQUIRE(!static_cast<bool>(p));
        pointer_type p1{&v};
        REQUIRE(static_cast<bool>(p1));
    }
    SECTION("cast_to_pointer_to_const_operator"){
        using test_cuda_memory::test_pointer;
        test_pointer<value_type> p{&v};
        auto cp = static_cast<test_pointer<const value_type>>(p);
        REQUIRE(p == cp);
    }
    SECTION("mutating_addition_operators"){
        value_type a[10];
        auto p = pointer_type{a};
        REQUIRE(++p == pointer_type{a+1});
        REQUIRE(p++ == pointer_type{a+1});
        REQUIRE(p == pointer_type{a+2});
        REQUIRE((p+=3) == pointer_type{a+5});
    }
    SECTION("mutating_subtraction_operators"){
        constexpr std::size_t n{10};
        value_type a[n];
        auto p = pointer_type{a+n};
        REQUIRE(--p == pointer_type{a+n-1});
        REQUIRE(p-- == pointer_type{a+n-1});
        REQUIRE(p == pointer_type{a+n-2});
        REQUIRE((p-=3) == pointer_type{a+n-5});
    }
    SECTION("comparisons"){
        constexpr std::size_t n{10};
        value_type a[n];
        auto p = pointer_type{a};
        REQUIRE(p > pointer_type{});
        REQUIRE(pointer_type{} < p);
        REQUIRE(pointer_type{a+1} > p);
        REQUIRE(p < pointer_type{a+1});
        REQUIRE(p <= p);
        REQUIRE(p >= p);
        REQUIRE(pointer_type{} >= pointer_type{});
        REQUIRE(pointer_type{} <= pointer_type{});
        REQUIRE(pointer_type{a+1} >= p);
        REQUIRE(p <= pointer_type{a+1});
    }
}

TEMPLATE_TEST_CASE("test_device_allocator","[test_cuda_memory]",
    float,
    (std::array<int,4>),
    (test_cuda_memory::test_array<test_cuda_memory::test_lin_space<double>,10>)
)
{
    using value_type = TestType;
    using allocator_type = cuda_experimental::device_allocator<value_type>;
    using cuda_experimental::cuda_get_device;

    std::size_t n{100};
    allocator_type allocator{};
    auto dev = cuda_get_device();
    auto ptr = allocator.allocate(n);
    REQUIRE(ptr.get() != nullptr);
    REQUIRE(ptr.device() == dev);
    allocator.deallocate(ptr,n);
}

TEMPLATE_TEST_CASE("test_locked_allocator","[test_cuda_memory]",
    float,
    (std::array<int,4>),
    (test_cuda_memory::test_array<test_cuda_memory::test_lin_space<double>,10>)
)
{
    using value_type = TestType;
    using allocator_type = cuda_experimental::locked_allocator<value_type>;

    std::size_t n{100};
    allocator_type allocator{};
    auto ptr = allocator.allocate(n);
    REQUIRE(ptr.get() != nullptr);
    allocator.deallocate(ptr,n);
}

TEMPLATE_TEST_CASE("test_registered_allocator","[test_cuda_memory]",
    float,
    (std::array<int,4>),
    (test_cuda_memory::test_array<test_cuda_memory::test_lin_space<double>,10>)
)
{
    using value_type = TestType;
    using allocator_type = cuda_experimental::registered_allocator<value_type>;

    std::size_t n{100};
    allocator_type allocator{};
    auto ptr = allocator.allocate(n);
    REQUIRE(ptr.get() != nullptr);
    allocator.deallocate(ptr,n);
}

TEST_CASE("test_align","[test_cuda_memory]"){
    using cuda_experimental::align;

    REQUIRE(align<1>(reinterpret_cast<void*>(1)) == reinterpret_cast<void*>(1));
    REQUIRE(align<1>(reinterpret_cast<void*>(100)) == reinterpret_cast<void*>(100));
    REQUIRE(align<2>(reinterpret_cast<void*>(1)) == reinterpret_cast<void*>(2));
    REQUIRE(align<2>(reinterpret_cast<void*>(100)) == reinterpret_cast<void*>(100));
    REQUIRE(align<4>(reinterpret_cast<void*>(1)) == reinterpret_cast<void*>(4));
    REQUIRE(align<4>(reinterpret_cast<void*>(100)) == reinterpret_cast<void*>(100));
    REQUIRE(align<8>(reinterpret_cast<void*>(1)) == reinterpret_cast<void*>(8));
    REQUIRE(align<8>(reinterpret_cast<void*>(32)) == reinterpret_cast<void*>(32));
    REQUIRE(align<8>(reinterpret_cast<void*>(100)) == reinterpret_cast<void*>(104));
    REQUIRE(align<32>(reinterpret_cast<void*>(1)) == reinterpret_cast<void*>(32));
    REQUIRE(align<32>(reinterpret_cast<void*>(88)) == reinterpret_cast<void*>(96));
    REQUIRE(align<32>(reinterpret_cast<void*>(1024)) == reinterpret_cast<void*>(1024));
}

TEST_CASE("test_alignment","[test_cuda_memory]"){
    using cuda_experimental::alignment;

    REQUIRE(alignment(reinterpret_cast<void*>(1)) == 1);
    REQUIRE(alignment(reinterpret_cast<void*>(2)) == 2);
    REQUIRE(alignment(reinterpret_cast<void*>(3)) == 1);
    REQUIRE(alignment(reinterpret_cast<void*>(4)) == 4);
    REQUIRE(alignment(reinterpret_cast<void*>(5)) == 1);
    REQUIRE(alignment(reinterpret_cast<void*>(8)) == 8);
    REQUIRE(alignment(reinterpret_cast<void*>(16)) == 16);
    REQUIRE(alignment(reinterpret_cast<void*>(32)) == 32);
    REQUIRE(alignment(reinterpret_cast<void*>(32*32)) == 1024);
}

TEST_CASE("test_align_for_copy","[test_cuda_memory]"){
    using cuda_experimental::aligned_for_copy;
    using block_type = __m256i;
    REQUIRE(sizeof(block_type) == 32);
    REQUIRE(alignof(block_type) == 32);
    using test_type = std::tuple<std::uintptr_t, std::size_t, std::uintptr_t, std::uintptr_t, std::ptrdiff_t, std::size_t, std::uintptr_t, std::uintptr_t, std::ptrdiff_t>;
    //0ptr, 1n, 2expected_first, 3expected_first_aligned, 4expected_first_offset, 5expected_aligned_blocks , 6expected_last, 7expected_last_aligned, 8expected_last_offset
    auto test_data = GENERATE(
        test_type{100, 300, 100, 128, 28, 8, 400, 384, 16},
        test_type{1000, 100, 1000, 1024, 24, 2, 1100, 1088, 12},
        test_type{1024, 100, 1024, 1024, 0, 3, 1124, 1120, 4},
        test_type{1024, 512, 1024, 1024, 0, 16, 1536, 1536, 0},
        test_type{100, 30, 100, 128, 28, 0, 130, 128, 2},
        test_type{112, 16, 112, 128, 16, 0, 128, 128, 0},
        test_type{100, 20, 100, 128, 28, 0, 120, 128, 0},
        test_type{100, 0, 100, 128, 28, 0, 100, 128, 0}
    );
    auto p = std::get<0>(test_data);
    auto n = std::get<1>(test_data);
    auto expected_first = std::get<2>(test_data);
    auto expected_first_aligned = std::get<3>(test_data);
    auto expected_first_offset = std::get<4>(test_data);
    auto expected_aligned_blocks = std::get<5>(test_data);
    auto expected_last = std::get<6>(test_data);
    auto expected_last_aligned = std::get<7>(test_data);
    auto expected_last_offset = std::get<8>(test_data);

    auto aligned = aligned_for_copy<block_type>(reinterpret_cast<void*>(p), n);
    REQUIRE(aligned.n() == n);
    REQUIRE(reinterpret_cast<std::uintptr_t>(aligned.first()) == expected_first);
    REQUIRE(reinterpret_cast<std::uintptr_t>(aligned.first_aligned()) == expected_first_aligned);
    REQUIRE(aligned.first_offset() == expected_first_offset);
    REQUIRE(aligned.aligned_blocks() == expected_aligned_blocks);
    REQUIRE(reinterpret_cast<std::uintptr_t>(aligned.last()) == expected_last);
    REQUIRE(reinterpret_cast<std::uintptr_t>(aligned.last_aligned()) == expected_last_aligned);
    REQUIRE(aligned.last_offset() == expected_last_offset);
}

// TEST_CASE("test_align_for_copy","[test_cuda_memory]"){
//     using cuda_experimental::align_for_copy;
//     using block_type = __m256i;
//     REQUIRE(sizeof(block_type) == 32);
//     REQUIRE(alignof(block_type) == 32);
//     using test_type = std::tuple<std::uintptr_t, std::uintptr_t, std::size_t, std::size_t, std::uintptr_t, std::uintptr_t, std::ptrdiff_t, std::ptrdiff_t, std::size_t, std::size_t,
//         std::uintptr_t, std::uintptr_t>;
//     //0src, 1dst, 2n_src, 3n_dst , 4expected_src_begin_aligned, 5expected_dst_begin_aligned, 6expected_src_begin_offset, 7expected_dst_begin_offset
//     //8expected_aligned_blocks_in_src , 9expected_aligned_blocks_in_dst, 10expected_src_end_aligned, 11expected_dst_end_aligned
//     auto test_data = GENERATE(
//         test_type{100, 1000, 300, 100, 128, 1024, 28, 24, 8, 2, 384, 1088}
//     );
//     auto src = std::get<0>(test_data);
//     auto dst = std::get<1>(test_data);
//     auto n_src = std::get<2>(test_data);
//     auto n_dst = std::get<3>(test_data);
//     auto expected_src_begin_aligned = std::get<4>(test_data);
//     auto expected_dst_begin_aligned = std::get<5>(test_data);
//     auto expected_src_begin_offset = std::get<6>(test_data);
//     auto expected_dst_begin_offset = std::get<7>(test_data);
//     auto expected_aligned_blocks_in_src = std::get<8>(test_data);
//     auto expected_aligned_blocks_in_dst = std::get<9>(test_data);
//     auto expected_src_end_aligned = std::get<10>(test_data);
//     auto expected_dst_end_aligned = std::get<11>(test_data);

//     auto aligned = align_for_copy<block_type>(reinterpret_cast<void*>(src), reinterpret_cast<void*>(dst), n_src, n_dst);
//     REQUIRE(reinterpret_cast<std::uintptr_t>(aligned.src_begin_aligned) == expected_src_begin_aligned);
//     REQUIRE(reinterpret_cast<std::uintptr_t>(aligned.dst_begin_aligned) == expected_dst_begin_aligned);
//     REQUIRE(aligned.src_begin_offset == expected_src_begin_offset);
//     REQUIRE(aligned.dst_begin_offset == expected_dst_begin_offset);
//     REQUIRE(aligned.aligned_blocks_in_src == expected_aligned_blocks_in_src);
//     REQUIRE(aligned.aligned_blocks_in_dst == expected_aligned_blocks_in_dst);
//     REQUIRE(reinterpret_cast<std::uintptr_t>(aligned.src_end_aligned) == expected_src_end_aligned);
//     REQUIRE(reinterpret_cast<std::uintptr_t>(aligned.dst_end_aligned) == expected_dst_end_aligned);
// }

// TEMPLATE_TEST_CASE("test_copy","[test_cuda_memory]",
//     cuda_experimental::device_allocator<float>,
//     (cuda_experimental::device_allocator<test_cuda_memory::test_array<std::size_t,5>>)
// )
// {
//     using allocator_type = TestType;
//     using value_type = typename allocator_type::value_type;
//     using cuda_experimental::copy;
//     using pointer_type = typename allocator_type::pointer;
//     using const_pointer_type = typename allocator_type::const_pointer;

//     auto allocator = allocator_type{};
//     constexpr std::size_t n{100};
//     auto dev_ptr = allocator.allocate(n);
//     auto const_dev_ptr = const_pointer_type{dev_ptr};
//     constexpr std::size_t a_len{10};
//     value_type a[a_len] = {value_type(1),value_type(2),value_type(3),value_type(4),value_type(5),value_type(6),value_type(7),value_type(8),value_type(9),value_type(10)};

//     SECTION("copy_host_device"){
//         value_type a_copy[a_len]{};
//         copy(a,a+a_len,dev_ptr);
//         SECTION("copy_from_dev_ptr"){
//             copy(dev_ptr,dev_ptr+a_len,a_copy);
//         }
//         SECTION("copy_from_const_dev_ptr"){
//             copy(const_dev_ptr,const_dev_ptr+a_len,a_copy);
//         }
//         REQUIRE(std::equal(a,a+a_len,a_copy));
//     }
//     SECTION("copy_host_device_iter"){
//         auto v = std::vector<value_type>(a,a+a_len);
//         auto v_copy = std::vector<value_type>(a_len);
//         copy(v.begin(),v.end(),dev_ptr);
//         SECTION("copy_from_dev_ptr"){
//             copy(dev_ptr,dev_ptr+a_len,v_copy.begin());
//         }
//         SECTION("copy_from_const_dev_ptr"){
//             copy(const_dev_ptr,const_dev_ptr+a_len,v_copy.begin());
//         }
//         REQUIRE(std::equal(v.begin(),v.end(),v_copy.begin()));
//     }
//     SECTION("copy_device_device"){
//         value_type a_copy[a_len]{};
//         copy(a,a+a_len,dev_ptr);
//         auto dev_ptr_copy = allocator.allocate(a_len);
//         SECTION("copy_from_dev_ptr"){
//             copy(dev_ptr,dev_ptr+a_len,dev_ptr_copy);
//             copy(dev_ptr_copy,dev_ptr_copy+a_len,a_copy);
//         }
//         SECTION("copy_from_const_dev_ptr"){
//             copy(const_dev_ptr,const_dev_ptr+a_len,dev_ptr_copy);
//             copy(const_dev_ptr,const_dev_ptr+a_len,a_copy);
//         }
//         REQUIRE(std::equal(a,a+a_len,a_copy));
//         allocator.deallocate(dev_ptr_copy,a_len);
//     }
//     allocator.deallocate(dev_ptr,n);
// }

// TEMPLATE_TEST_CASE("test_fill","[test_cuda_memory]",
//     cuda_experimental::device_allocator<float>
// )
// {
//     using allocator_type = TestType;
//     using value_type = typename allocator_type::value_type;
//     using cuda_experimental::fill;
//     using pointer_type = typename allocator_type::pointer;
//     using const_pointer_type = typename allocator_type::const_pointer;

//     auto allocator = allocator_type{};
//     constexpr std::size_t n{100};
//     auto dev_ptr = allocator.allocate(n);
//     value_type v{11};
//     fill(dev_ptr, dev_ptr+n, v);
//     std::vector<value_type> dev_copy(n);
//     copy(dev_ptr, dev_ptr+n, dev_copy.begin());
//     std::vector<value_type> expected(n,v);
//     REQUIRE(dev_copy == expected);
//     allocator.deallocate(dev_ptr,n);
// }

// TEST_CASE("test_device_pointer","[test_cuda_memory]"){
//     using value_type = float;
//     using allocator_type = cuda_experimental::device_allocator<value_type>;
//     using cuda_experimental::cuda_get_device;

//     allocator_type allocator{};
//     std::vector<value_type> v{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
//     auto n = v.size();
//     auto device = cuda_get_device();
//     auto ptr_dev = allocator.allocate(n);
//     REQUIRE(ptr_dev.device() == device);
//     copy(v.begin(), v.end(), ptr_dev);

//     SECTION("device_pointer_dereference"){
//         auto it = ptr_dev;
//         auto end = ptr_dev+n;
//         auto const_it = ptr_to_const(it);
//         SECTION("read_dev_reference"){
//             std::vector<value_type> v_dev_copy{};
//             std::vector<value_type> v_const_dev_copy{};
//             for(;it!=end; ++it, ++const_it){
//                 v_dev_copy.push_back(*it);
//                 v_const_dev_copy.push_back(*const_it);
//             }
//             REQUIRE(std::equal(v.begin(),v.end(),v_dev_copy.begin()));
//             REQUIRE(std::equal(v.begin(),v.end(),v_const_dev_copy.begin()));
//         }
//         SECTION("write_dev_reference"){
//             std::vector<value_type> v_expected_result{2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40};
//             std::vector<value_type> v_expected_result_{};
//             for(;it!=end; ++it){
//                 auto v = *it*2;
//                 *it = v;
//             }
//             std::vector<value_type> v_dev_copy(n);
//             copy(ptr_dev, ptr_dev+n, v_dev_copy.begin());
//             REQUIRE(std::equal(v_expected_result.begin(),v_expected_result.end(),v_dev_copy.begin()));
//         }
//     }
//     SECTION("device_pointer_subscription"){
//         SECTION("read_dev_reference"){
//             std::vector<value_type> v_dev_copy{};
//             std::vector<value_type> v_const_dev_copy{};
//             auto ptr_const_dev = ptr_to_const(ptr_dev);
//             for(std::size_t i{0};i!=n; ++i){
//                 v_dev_copy.push_back(ptr_dev[i]);
//                 v_const_dev_copy.push_back(ptr_const_dev[i]);
//             }
//             REQUIRE(std::equal(v.begin(),v.end(),v_dev_copy.begin()));
//             REQUIRE(std::equal(v.begin(),v.end(),v_const_dev_copy.begin()));
//         }
//         SECTION("write_dev_reference"){
//             std::vector<value_type> v_expected_result{};
//             for(std::size_t i{0}; i!=n; ++i){
//                 auto v = i%2;
//                 ptr_dev[i] = v;
//                 v_expected_result.push_back(v);
//             }
//             std::vector<value_type> v_dev_copy(n);
//             copy(ptr_dev, ptr_dev+n, v_dev_copy.begin());
//             REQUIRE(std::equal(v_expected_result.begin(),v_expected_result.end(),v_dev_copy.begin()));
//         }
//     }
//     SECTION("device_pointer_iteration"){
//         auto begin = ptr_dev;
//         auto end = ptr_dev+n;
//         SECTION("read"){
//             auto cbegin = ptr_to_const(begin);
//             auto cend = ptr_to_const(end);
//             REQUIRE(std::equal(begin, end, v.begin()));
//             REQUIRE(std::equal(cbegin, cend, v.begin()));
//         }
//         SECTION("write"){
//             auto transformed_begin = allocator.allocate(n);
//             auto transformator = [](const auto& v){return v+1;};
//             std::vector<value_type> v_transformed{2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21};
//             std::transform(begin,end,transformed_begin,transformator);

//             REQUIRE(std::equal(v_transformed.begin(),v_transformed.end(), transformed_begin));
//             allocator.deallocate(transformed_begin,n);
//         }
//     }

//     allocator.deallocate(ptr_dev, n);
// }

// TEST_CASE("test_locked_pointer","[test_cuda_memory]"){
//     using value_type = float;
//     using allocator_type = cuda_experimental::locked_allocator<value_type>;
//     using difference_type = typename allocator_type::difference_type;

//     allocator_type allocator{};
//     std::vector<value_type> v{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
//     auto n = v.size();
//     auto ptr_locked = allocator.allocate(n);
//     REQUIRE(std::is_same_v<decltype(*ptr_locked),value_type&>);
//     REQUIRE(std::is_same_v<decltype(*ptr_to_const(ptr_locked)),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(ptr_locked[std::declval<difference_type>()]),value_type&>);
//     REQUIRE(std::is_same_v<decltype(ptr_to_const(ptr_locked)[std::declval<difference_type>()]),const value_type&>);
//     std::copy(v.begin(), v.end(), ptr_locked);
//     REQUIRE(std::equal(ptr_locked, ptr_locked+n, v.begin()));
//     REQUIRE(std::equal(ptr_to_const(ptr_locked), ptr_to_const(ptr_locked+n), v.begin()));

//     auto transformed = allocator.allocate(n);
//     auto transformator = [](const auto& v){return v+1;};
//     std::vector<value_type> v_expected{2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21};
//     std::transform(ptr_locked,ptr_locked+n,transformed,transformator);
//     REQUIRE(std::equal(v_expected.begin(),v_expected.end(), transformed));
//     allocator.deallocate(transformed,n);

//     allocator.deallocate(ptr_locked, n);
// }
