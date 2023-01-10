#include <iostream>
#include "catch.hpp"

namespace experimental_ptr_alloc_copy{

template<typename T>
class device_pointer{
    int device_{0};
    T* ptr{};
public:
    auto get()const{return ptr;}
    auto device()const{return device_;}
    auto operator*(){
        copy(*this,*this, 0);
        return T{};
    }
};

class ttt{

    auto operator()(){
        //copy(device_pointer<float>{}, device_pointer<float>{}, 0);
    }
};

// void h(){
//     copy(device_pointer<float>{}, device_pointer<float>{}, 0);
// }

template<typename T>
class device_allocator{
public:
    using pointer = device_pointer<T>;
    auto allocate(std::size_t n){
        return pointer{};
    }
    auto deallocate(pointer ptr, std::size_t n){
    }

};

template<typename T>
void copy(device_pointer<T> dst, device_pointer<T> src, std::size_t n){
    if (dst.device() == src.device()){
        std::cout<<std::endl<<"same_device_copy"<<dst.get()<<src.get();
    }else{
        std::cout<<std::endl<<"inter_device_copy"<<dst.get()<<src.get();
    }
}

void f(){
    device_allocator<float> alloc{};
    auto src = alloc.allocate(0);
    auto dst = alloc.allocate(0);
    auto res = *src;
    copy(dst,src,0);
}

}   //end of namespace experimental_ptr_alloc_copy


TEST_CASE("experimental_ptr_alloc_copy","[experimental_ptr_alloc_copy]"){
    using experimental_ptr_alloc_copy::f;
    using experimental_ptr_alloc_copy::copy;
    using experimental_ptr_alloc_copy::device_allocator;
    using experimental_ptr_alloc_copy::device_pointer;
    using value_type = float;

    f();
}