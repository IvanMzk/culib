/*
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#ifndef CUDA_STORAGE_HPP_
#define CUDA_STORAGE_HPP_

#include <memory>
#include <iostream>
#include "cuda_algorithm.hpp"

namespace culib{

namespace detail{

template<typename, typename = void> constexpr bool is_iterator = false;
template<typename T> constexpr bool is_iterator<T,std::void_t<typename std::iterator_traits<T>::iterator_category>> = true;

template<typename Alloc>
struct row_buffer
{
    using allocator_type = Alloc;
    using pointer = typename std::allocator_traits<Alloc>::pointer;
    using size_type = typename std::allocator_traits<Alloc>::size_type;

    allocator_type& allocator_;
    size_type size_;
    pointer ptr_;
    ~row_buffer(){
        if (ptr_){
            allocator_.deallocate(ptr_,size_);
        }
    }

    row_buffer() = default;
    row_buffer(const row_buffer&) = delete;
    row_buffer(row_buffer&&) = delete;
    row_buffer(allocator_type& allocator__, size_type size__, pointer ptr__):
        allocator_{allocator__},
        size_{size__},
        ptr_{ptr__}
    {}

    allocator_type& get_allocator(){
        return allocator_;
    }
    pointer get()const{
        return ptr_;
    }
    pointer release(){
        auto res = ptr_;
        ptr_ = nullptr;
        return res;
    }
};

}   //end of namespace detail

/*
* cuda_storage manages memory block
* cuda_storage use cuda_aware_allocator to allocate memory
* memory block may reside on device , on host or be UM memory block, it depends on allocator type
* allocation is made with respect to current active device of calling thread, allocator must gurantee such allocation semantic
* to construct or copy storage that reside on specific device user must use cuda api to set device
* move operations are guaranteed allocation free
* iterator and pointer returned by data() may not be dereferenceable on device, it depends on allocator
* iterator is guaranteed dereferenceable on host, but not efficient for device memory
*/
template<typename T, typename Alloc = device_allocator<T>>
class cuda_storage
{
    static_assert(std::is_trivially_copyable_v<T>);
public:
    using allocator_type = Alloc;
    using value_type = T;
    using pointer = typename allocator_type::pointer;
    using const_pointer = typename allocator_type::const_pointer;
    using iterator = pointer;
    using const_iterator = const_pointer;
    using difference_type = typename allocator_type::difference_type;
    using size_type = typename allocator_type::size_type;

    ~cuda_storage()
    {
        free();
    }
    //default constructor, no allocation take place
    explicit cuda_storage(const allocator_type& alloc = allocator_type()):
        allocator_{alloc}
    {}
    //reallocate if not equal sizes or not equal allocators
    cuda_storage& operator=(const cuda_storage& other){
        if (this != &other){
            copy_assign(other, typename std::allocator_traits<allocator_type>::propagate_on_container_copy_assignment());
        }
        return *this;
    }
    //use copy assignment if other's allocator disallow to propagate and allocators not equal, otherwise steal from other and put other in default state
    cuda_storage& operator=(cuda_storage&& other){
        if (this != &other){
            move_assign(std::move(other),  typename std::allocator_traits<allocator_type>::propagate_on_container_move_assignment());
        }
        return *this;
    }
    //no reallocation guarantee
    cuda_storage(cuda_storage&& other):
        allocator_{std::move(other.allocator_)},
        begin_{other.begin_},
        end_{other.end_}
    {
        other.begin_ = nullptr;
        other.end_ = nullptr;
    }
    //construct storage with n uninitialized elements
    explicit cuda_storage(const size_type& n, const allocator_type& alloc = allocator_type()):
        allocator_{alloc},
        begin_{allocator_.allocate(n)},
        end_{begin_+n}
    {}
    //construct storage with n elements initialized with v
    cuda_storage(const size_type& n, const value_type& v, const allocator_type& alloc = allocator_type()):
        allocator_{alloc}
    {
        init(n,v);
    }
    //construct storage from iterators range
    template<typename It, std::enable_if_t<detail::is_iterator<It>,int> =0 >
    cuda_storage(It first, It last, const allocator_type& alloc = allocator_type()):
        allocator_{alloc}
    {
        init(first,last);
    }
    //construct storage from host init list
    cuda_storage(std::initializer_list<value_type> init_data, const allocator_type& alloc = allocator_type()):
        allocator_{alloc}
    {
        init(init_data.begin(),init_data.end());
    }

    value_type* data(){return static_cast<value_type*>(begin_);}
    const value_type* data()const{return  static_cast<const value_type*>(begin_);}
    iterator begin(){return begin_;}
    iterator end(){return end_;}
    const_iterator begin()const{return static_cast<const_pointer>(begin_);}
    const_iterator end()const{return  static_cast<const_pointer>(end_);}
    size_type size()const{return end_-begin_;}
    bool empty()const{return begin()==end();}
    cuda_storage clone()const{return cuda_storage{*this};}
    void clear(){free();}
    allocator_type get_allocator()const{return allocator_;}

private:
    //private copy constructor, use clone() to make copy, if storage allocates on device it use current active device of calling thread
    cuda_storage(const cuda_storage& other):
        allocator_{std::allocator_traits<allocator_type>::select_on_container_copy_construction(other.get_allocator())}
    {
        init(other.begin(),other.end());
    }

    //no copy assign other's allocator
    void copy_assign(const cuda_storage& other, std::false_type){
        auto other_size = other.size();
        if (size()!=other_size){
            auto new_buffer = allocate_buffer(other_size);
            copy(other.begin(),other.end(),new_buffer.get());
            free();
            begin_ = new_buffer.release();
            end_ = begin_+other_size;
        }else{
            copy(other.begin(),other.end(),begin_);
        }
    }

    //copy assign other's allocator
    void copy_assign(const cuda_storage& other, std::true_type){
        if (allocator_ ==  other.allocator_ || typename std::allocator_traits<allocator_type>::is_always_equal()){
            copy_assign(other, std::false_type{});
        }else{
            auto other_size = other.size();
            auto other_allocator = other.get_allocator();
            auto new_buffer = allocate_buffer(other_size,other_allocator);
            copy(other.begin(),other.end(),new_buffer.get());
            auto old_alloc = std::move(allocator_);
            allocator_ = std::move(other_allocator);
            free(old_alloc);
            begin_ = new_buffer.release();
            end_ = begin_+other_size;
        }
    }

    //no move assign other's allocator, if allocators not equal copy are made
    void move_assign(cuda_storage&& other, std::false_type){
        if (allocator_ ==  other.allocator_ || typename std::allocator_traits<allocator_type>::is_always_equal()){
            free();
            begin_ = other.begin_;
            end_ = other.end_;
            other.begin_ = nullptr;
            other.end_ = nullptr;
        }else{
            copy_assign(other, std::false_type{});
        }
    }

    //move assign other's allocator
    void move_assign(cuda_storage&& other, std::true_type){
        auto old_alloc = std::move(allocator_);
        allocator_ = std::move(other.allocator_);
        free(old_alloc);
        begin_ = other.begin_;
        end_ = other.end_;
        other.begin_ = nullptr;
        other.end_ = nullptr;
    }

    void init(const size_type& n, const value_type& v){
        auto buf = allocate_buffer(n);
        fill(buf.get(), buf.get()+n, v);
        begin_=buf.release();
        end_=begin_+n;
    }

    template<typename It>
    void init(It first, It last){
        const auto n = static_cast<size_type>(std::distance(first,last));
        auto buf = allocate_buffer(n);
        copy(first,last,buf.get());
        begin_=buf.release();
        end_=begin_+n;
    }

    auto allocate_buffer(const size_type& n, allocator_type& alloc){
        return detail::row_buffer<allocator_type>{alloc,n,alloc.allocate(n)};
    }
    auto allocate_buffer(const size_type& n){
        return allocate_buffer(n,allocator_);
    }

    //destroy and deallocate
    void free(allocator_type& alloc){
        if (begin_){
            alloc.deallocate(begin_,size());
            begin_ = nullptr;
            end_ = nullptr;
        }
    }
    void free(){
        free(allocator_);
    }

    allocator_type allocator_;
    pointer begin_{nullptr};
    pointer end_{nullptr};
};

}   //end of namespace culib
#endif