#ifndef DEVICE_STORAGE_HPP_
#define DEVICE_STORAGE_HPP_

#include <memory>
#include <iostream>
#include "cuda_memory.hpp"

namespace cuda_experimental{

namespace detail{
    template<typename, typename = void> constexpr bool is_iterator = false;
    template<typename T> constexpr bool is_iterator<T,std::void_t<typename std::iterator_traits<T>::iterator_category>> = true;
}   //end of namespace detail


template<typename T, typename Alloc = cuda_allocator<T>>
class device_storage : private Alloc
{
    using allocator_type = Alloc;
public:
    using difference_type = typename allocator_type::difference_type;
    using size_type = typename allocator_type::size_type;
    using value_type = typename allocator_type::value_type;
    using pointer = typename allocator_type::pointer;
    using const_pointer = typename allocator_type::const_pointer;
    static_assert(std::is_trivially_copyable_v<value_type>);

    ~device_storage(){deallocate();}
    device_storage& operator=(const device_storage&) = delete;
    device_storage& operator=(device_storage&&) = delete;
    device_storage() = default;
    device_storage(device_storage&& other):
        size_{other.size_},
        begin_{other.begin_}
    {
        other.size_ = 0;
        other.begin_ = nullptr;
    }
    //construct storage with n uninitialized elements
    explicit device_storage(const size_type& n):
        size_{n},
        begin_{allocate(n)}
    {}
    //construct storage with n elements initialized with v
    device_storage(const size_type& n, const value_type& v):
        size_{n},
        begin_{allocate(n)}
    {
        auto buffer = make_host_buffer<value_type>(n);
        std::uninitialized_fill_n(buffer.get(),n,v);
        copy(buffer.get(), buffer.get()+n, begin_);
    }
    //construct storage from host iterators range
    template<typename It, std::enable_if_t<detail::is_iterator<It> ,int> =0 >
    device_storage(It first, It last):
        size_{std::distance(first,last)},
        begin_{allocate(size_)}
    {
        copy(first,last,begin_);
    }
    //construct storage from host init list
    device_storage(std::initializer_list<value_type> init_data):
        device_storage(init_data.begin(), init_data.end())
    {}
    //construct storage from device iterators range
    device_storage(const_pointer first, const_pointer last):
        size_{distance(first,last)},
        begin_{allocate(size_)}
    {
        copy(first,last,begin_);
    }

    auto data(){return begin_;}
    auto data()const{return  const_pointer(begin_);}
    auto device_begin(){return begin_;}
    auto device_end(){return  begin_ + size_;}
    auto device_begin()const{return const_pointer{begin_};}
    auto device_end()const{return  const_pointer{begin_ + size_};}
    auto size()const{return size_;}
    auto empty()const{return !static_cast<bool>(begin_);}
    auto clone()const{return device_storage{*this};}
    void free(){deallocate();}

private:
    //private copy constructor, should use clone() to make copy
    device_storage(const device_storage& other):
        device_storage(other.device_begin(),other.device_end())
    {}
    pointer allocate(const size_type& n){
        return allocator_type::allocate(n);
    }
    void deallocate(){
        if (begin_){
            allocator_type::deallocate(begin_,size_);
            size_ = 0;
            begin_ = nullptr;
        }
    }

    size_type size_;
    pointer begin_;
};

}   //end of namespace cuda_experimental



#endif