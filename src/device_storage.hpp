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

/*
* allocator must have semantic:
* Alloc a{};
* a allocate on device active at the moment of a constructing
* Alloc a{device_id};
* a allocate on device with device_id
* Alloc a1{a};
* Alloc a2{std::move(a)};
* a1, a2 must allocate on same device as a, that is allocator must have state
*/
template<typename T, typename Alloc = cuda_allocator<T>>
class device_storage
{
public:
    using allocator_type = Alloc;
    using difference_type = typename allocator_type::difference_type;
    using size_type = typename allocator_type::size_type;
    using value_type = typename allocator_type::value_type;
    using pointer = typename allocator_type::pointer;
    using const_pointer = typename allocator_type::const_pointer;
    static_assert(std::is_trivially_copyable_v<value_type>);
    static_assert(!std::allocator_traits<allocator_type>::is_always_equal());
    static_assert(std::allocator_traits<allocator_type>::propagate_on_container_copy_assignment());
    static_assert(std::allocator_traits<allocator_type>::propagate_on_container_move_assignment());

    ~device_storage(){deallocate();}
    device_storage(const allocator_type& alloc = allocator_type()):
        allocator_{alloc},
        size_{0},
        begin_{nullptr}
    {}
    //reallocate if not equal sizes or not equal allocators, result is always on device that other is
    device_storage& operator=(const device_storage& other){
        copy_assign(other);
        return *this;
    }
    //no reallocation guarantee, result is always on device that other is
    device_storage& operator=(device_storage&& other){
        move_assign(std::move(other));
        return *this;
    }
    //no reallocation guarantee, result is always on device that other is
    device_storage(device_storage&& other):
        allocator_{std::move(other.allocator_)},
        size_{other.size_},
        begin_{other.begin_}
    {
        other.size_ = 0;
        other.begin_ = nullptr;
    }
    //construct storage with n uninitialized elements
    explicit device_storage(const size_type& n, const allocator_type& alloc = allocator_type()):
        allocator_{alloc},
        size_{n},
        begin_{allocate(n)}
    {}
    //construct storage with n elements initialized with v
    device_storage(const size_type& n, const value_type& v, const allocator_type& alloc = allocator_type()):
        allocator_{alloc},
        size_{n},
        begin_{allocate(n)}
    {
        auto buffer = make_host_buffer<value_type>(n);
        std::uninitialized_fill_n(buffer.get(),n,v);
        copy(buffer.get(), buffer.get()+n, begin_);
    }
    //construct storage from host iterators range
    template<typename It, std::enable_if_t<detail::is_iterator<It>,int> =0 >
    device_storage(It first, It last, const allocator_type& alloc = allocator_type()):
        allocator_{alloc},
        size_{std::distance(first,last)},
        begin_{allocate(size_)}
    {
        copy(first,last,begin_);
    }
    //construct storage from host init list
    device_storage(std::initializer_list<value_type> init_data, const allocator_type& alloc = allocator_type()):
        allocator_{alloc},
        size_{static_cast<size_type>(init_data.size())},
        begin_{allocate(size_)}
    {
        copy(init_data.begin(),init_data.end(),begin_);
    }
    //construct storage from device iterators range, src and dst must be allocated on same device
    device_storage(const_pointer first, const_pointer last, const allocator_type& alloc = allocator_type()):
        allocator_{alloc},
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
    auto get_allocator()const{return allocator_;}

private:
    //private copy constructor, use clone() to make copy
    //make copy on device that is other on
    device_storage(const device_storage& other):
        allocator_{std::allocator_traits<allocator_type>::select_on_container_copy_construction(other.get_allocator())},
        size_(other.size_),
        begin_(allocate(size_))
    {
        copy(other.device_begin(),other.device_end(),begin_);
    }
    //copy assign other allocator if allocator not equal
    void copy_assign(const device_storage& other){
        auto other_size = other.size();
        if (allocator_ ==  other.allocator_){
            if (size()!=other_size){
                auto new_buffer = allocate(other_size);
                deallocate();
                size_ = other_size;
                begin_ = new_buffer;
            }
        }else{
            auto other_allocator = other.get_allocator();
            auto new_buffer = other_allocator.allocate(other_size);
            deallocate();
            size_ = other_size;
            begin_ = new_buffer;
            allocator_ = other_allocator;
        }
        copy(other.device_begin(),other.device_end(),begin_);
    }
    //move assign other allocator
    void move_assign(device_storage&& other){
        deallocate();
        allocator_ = std::move(other.allocator_);
        size_ = other.size_;
        begin_ = other.begin_;
        other.size_ = 0;
        other.begin_ = nullptr;
    }

    pointer allocate(const size_type& n){
        return allocator_.allocate(n);
    }
    void deallocate(){
        if (begin_){
            allocator_.deallocate(begin_,size_);
            size_ = 0;
            begin_ = nullptr;
        }
    }

    allocator_type allocator_;
    size_type size_;
    pointer begin_;
};

}   //end of namespace cuda_experimental



#endif