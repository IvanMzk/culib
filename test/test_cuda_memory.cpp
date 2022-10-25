#include <iostream>
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
    using basic_pointer::operator=;
    test_pointer(pointer p = nullptr):
        basic_pointer(p)
    {}
};

template<typename D>
class A
{
public:
    A& operator=(const A&) = default;
    A& operator=(A&&) = default;
    D& operator=(std::nullptr_t){
        a = 0;
        return static_cast<D&>(*this);
    }
    D& operator=(int a_){
        a = a_;
        return static_cast<D&>(*this);

    }
    auto get(){return a;}
private:
    friend D;
    A(const A&) = default;
    A(A&&) = default;
    A(int a_ = 0):
        a{a_}
    {}
    int a;
};

class B : public A<B>
{
public:
    B(int b = 0) :
        A(b)
    {}
    using A::operator=;
};

}   //end of namespace test_cuda_memory

TEMPLATE_TEST_CASE("test_basic_pointer","[test_cuda_memory]",
    test_cuda_memory::test_pointer<float>,
    test_cuda_memory::test_pointer<const float>,
    cuda_experimental::device_pointer<float>,
    cuda_experimental::device_pointer<const float>
)
{

    SECTION("test_AB"){
        using test_cuda_memory::A;
        using test_cuda_memory::B;
        REQUIRE(std::is_trivially_copyable_v<A<B>>);
        REQUIRE(std::is_trivially_copyable_v<B>);
        B b{1};
        B b_{0};
        B bb{};
        REQUIRE(std::is_same_v<decltype(b_=b),B&>);
        b_ = b;
        REQUIRE(b_.get() == 1);
        REQUIRE(b.get() == 1);

        REQUIRE(std::is_same_v<decltype(b=nullptr),B&>);
        b = nullptr;
        REQUIRE(b.get() == 0);

        REQUIRE(std::is_same_v<decltype(b=1),B&>);
        b = 1;
        REQUIRE(b.get() == 1);
    }

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
    SECTION("operator++"){
        value_type a[10];
        auto p = pointer_type{a};
        REQUIRE(++p == pointer_type{a+1});
        REQUIRE(p++ == pointer_type{a+1});
        REQUIRE(p == pointer_type{a+2});
    }
    SECTION("operator--"){
        constexpr std::size_t n{10};
        value_type a[n];
        auto p = pointer_type{a+n};
        REQUIRE(--p == pointer_type{a+n-1});
        REQUIRE(p-- == pointer_type{a+n-1});
        REQUIRE(p == pointer_type{a+n-2});
    }
}

TEMPLATE_TEST_CASE("test_copy","[test_cuda_memory]",
    cuda_experimental::device_allocator<float>
)
{
    using value_type = float;
    using cuda_experimental::copy;
    using allocator_type = TestType;
    using pointer_type = typename allocator_type::pointer;
    using const_pointer_type = typename allocator_type::const_pointer;

    auto allocator = allocator_type{};
    constexpr std::size_t n{100};
    auto dev_ptr = allocator.allocate(n);
    auto const_dev_ptr = const_pointer_type{dev_ptr};

    SECTION("copy_host_device"){
        constexpr std::size_t a_len{10};
        value_type a[a_len] = {1,2,3,4,5,6,7,8,9,10};
        value_type a_copy[a_len]{};
        copy(a,a+a_len,dev_ptr);
        SECTION("copy_from_dev_ptr"){
            copy(dev_ptr,dev_ptr+a_len,a_copy);
        }
        SECTION("copy_from_const_dev_ptr"){
            copy(const_dev_ptr,const_dev_ptr+a_len,a_copy);
        }
        REQUIRE(std::equal(a,a+a_len,a_copy));
    }
    SECTION("copy_host_device_iter"){
        auto a = std::vector<value_type>{1,2,3,4,5,6,7,8,9,10};
        auto a_len = a.size();
        auto a_copy = std::vector<value_type>(a_len);
        copy(a.begin(),a.end(),dev_ptr);
        SECTION("copy_from_dev_ptr"){
            copy(dev_ptr,dev_ptr+a_len,a_copy.begin());
        }
        SECTION("copy_from_const_dev_ptr"){
            copy(const_dev_ptr,const_dev_ptr+a_len,a_copy.begin());
        }
        REQUIRE(std::equal(a.begin(),a.end(),a_copy.begin()));
    }
    SECTION("copy_device_device"){
        constexpr std::size_t a_len{10};
        value_type a[a_len] = {1,2,3,4,5,6,7,8,9,10};
        value_type a_copy[a_len]{};
        copy(a,a+a_len,dev_ptr);
        auto dev_ptr_copy = allocator.allocate(a_len);
        SECTION("copy_from_dev_ptr"){
            copy(dev_ptr,dev_ptr+a_len,dev_ptr_copy);
            copy(dev_ptr_copy,dev_ptr_copy+a_len,a_copy);
        }
        SECTION("copy_from_const_dev_ptr"){
            copy(const_dev_ptr,const_dev_ptr+a_len,dev_ptr_copy);
            copy(const_dev_ptr,const_dev_ptr+a_len,a_copy);
        }
        REQUIRE(std::equal(a,a+a_len,a_copy));
        allocator.deallocate(dev_ptr_copy,a_len);
    }
    allocator.deallocate(dev_ptr,n);
}

TEST_CASE("test_device_pointer","[test_cuda_memory]"){

    SECTION("vector_of_bools"){
        auto v = std::vector<bool>{1,0,0,0,1};
        auto it = v.begin();

        bool ee = *it;
        auto r = *it;

        *it = false;
        r = true;
        auto& cv = v;
        auto cit = cv.cbegin();
        //*cit = true;
        auto cr = *cit;
        cr = false;
    }

    using value_type = float;
    using allocator_type = cuda_experimental::device_allocator<value_type>;

    allocator_type allocator{};
    std::vector<value_type> v{1,2,3,4,5,6,7,8,9,10};
    auto n = v.size();
    auto ptr_dev = allocator.allocate(n);
    auto ptr_const_dev = ptr_to_const(ptr_dev);
    copy(v.begin(), v.end(), ptr_dev);


    auto r = *ptr_dev;
    REQUIRE(r == 1);
    auto val = *ptr_const_dev;
    REQUIRE(val == 1);

    allocator.deallocate(ptr_dev, n);
}


// TEST_CASE("test_pointer_attributes","[test_cuda_memory]"){
//     using value_type = float;
//     using cuda_mapping_allocator_type = cuda_experimental::cuda_mapping_allocator<value_type>;
//     using cuda_allocator_type = cuda_experimental::cuda_allocator<value_type>;
//     using cuda_experimental::unified_memory_allocator;
//     using cuda_experimental::basic_pointer;
//     using cuda_experimental::copy;
//     using cuda_experimental::cuda_assert;
//     using cuda_experimental::make_host_buffer;

// // enum __device_builtin__ cudaMemoryType
// // {
// //     cudaMemoryTypeUnregistered = 0, /**< Unregistered memory */
// //     cudaMemoryTypeHost         = 1, /**< Host memory */
// //     cudaMemoryTypeDevice       = 2, /**< Device memory */
// //     cudaMemoryTypeManaged      = 3  /**< Managed memory */
// // };

//     auto print_ptr_attr = [](const auto& p){
//         cudaPointerAttributes attr;
//         cuda_error_check(cudaPointerGetAttributes(&attr, p));
//         std::cout<<std::endl<<"device"<<attr.device;
//         std::cout<<std::endl<<"device_ptr"<<attr.devicePointer;
//         std::cout<<std::endl<<"host_ptr"<<attr.hostPointer;
//         switch (attr.type){
//             case cudaMemoryType::cudaMemoryTypeUnregistered:
//                 std::cout<<std::endl<<"Unregistered memory"<<attr.type;
//                 break;
//             case cudaMemoryType::cudaMemoryTypeHost:
//                 std::cout<<std::endl<<"Host memory"<<attr.type;
//                 break;
//             case cudaMemoryType::cudaMemoryTypeDevice:
//                 std::cout<<std::endl<<"Device memory"<<attr.type;
//                 break;
//             case cudaMemoryType::cudaMemoryTypeManaged:
//                 std::cout<<std::endl<<"Managed memory"<<attr.type;
//                 break;
//         }
//     };
//     auto print_cuda_ptr_attr = [&](const auto& p){
//         print_ptr_attr(p.get());
//     };

//     int n{100};
//     int offset{99};
//     auto mapping_alloc = cuda_mapping_allocator_type{};
//     auto p = mapping_alloc.allocate(n);
//     print_cuda_ptr_attr(p+offset);

//     auto buffer_to_register = make_host_buffer<value_type>(n);
//     auto mapping_alloc_registered = cuda_mapping_allocator_type{buffer_to_register.get()};
//     auto p_registered = mapping_alloc_registered.allocate(n);
//     print_cuda_ptr_attr(p_registered+offset);

//     auto dev_alloc = cuda_allocator_type{};
//     auto p_dev = dev_alloc.allocate(n);
//     print_cuda_ptr_attr(p_dev+offset);

//     auto buffer = make_host_buffer<value_type>(n);
//     //print_ptr_attr(buffer.get()+offset);

//     auto um_alloc = unified_memory_allocator<value_type>{};
//     auto um_ptr = um_alloc.allocate(n);
//     print_cuda_ptr_attr(um_ptr+offset);

//     um_alloc.deallocate(um_ptr,n);
//     mapping_alloc.deallocate(p,n);
//     mapping_alloc_registered.deallocate(p_registered,n);
//     dev_alloc.deallocate(p_dev,n);

// }