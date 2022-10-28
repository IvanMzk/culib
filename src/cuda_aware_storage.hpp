#ifndef CUDA_AWARE_STORAGE_HPP_
#define CUDA_AWARE_STORAGE_HPP_

#include <memory>
#include <iostream>
#include "cuda_memory.hpp"

namespace cuda_experimental{

namespace detail{
    template<typename, typename = void> constexpr bool is_iterator = false;
    template<typename T> constexpr bool is_iterator<T,std::void_t<typename std::iterator_traits<T>::iterator_category>> = true;
}   //end of namespace detail

/*
* cuda_aware_storage manages memory block
* cuda_aware_storage use cuda_aware_allocator to allocate memory
* memory block may reside on device , on host or be UM memory block, it depends on allocator type
* allocation is made with respect to current active device of calling thread, allocator must gurantee such allocation semantic
* to construct or copy storage that reside on specific device user must use cuda api to set device
* move operations are guaranteed allocation free
* iterator and pointer returned by data() may not be dereferenceable on device, it depends on allocator
* iterator is guaranteed dereferenceable on host, but not efficient for device memory
*/
template<typename T, typename Alloc = device_allocator<T>>
class cuda_aware_storage
{
public:
    using allocator_type = Alloc;
    using difference_type = typename allocator_type::difference_type;
    using size_type = typename allocator_type::size_type;
    using value_type = typename allocator_type::value_type;
    using pointer = typename allocator_type::pointer;
    using const_pointer = typename allocator_type::const_pointer;

    ~cuda_aware_storage(){deallocate();}
    //default constructor, no allocation take place
    cuda_aware_storage(const allocator_type& alloc = allocator_type()):
        allocator_{alloc},
        size_{0},
        begin_{nullptr}
    {}
    //reallocate if not equal sizes or not equal allocators
    cuda_aware_storage& operator=(const cuda_aware_storage& other){
        if (this != &other){
            copy_assign(other, typename std::allocator_traits<allocator_type>::propagate_on_container_copy_assignment());
        }
        return *this;
    }
    //use copy assignment if other's allocator disallow to propagate and allocators not equal, otherwise steal from other and put other in default state
    cuda_aware_storage& operator=(cuda_aware_storage&& other){
        if (this != &other){
            move_assign(std::move(other),  typename std::allocator_traits<allocator_type>::propagate_on_container_move_assignment());
        }
        return *this;
    }
    //no reallocation guarantee
    cuda_aware_storage(cuda_aware_storage&& other):
        allocator_{std::move(other.allocator_)},
        size_{other.size_},
        begin_{other.begin_}
    {
        other.size_ = 0;
        other.begin_ = nullptr;
    }
    //construct storage with n uninitialized elements
    explicit cuda_aware_storage(const size_type& n, const allocator_type& alloc = allocator_type()):
        allocator_{alloc},
        size_{n},
        begin_{allocate(n)}
    {}
    //construct storage with n elements initialized with v
    cuda_aware_storage(const size_type& n, const value_type& v, const allocator_type& alloc = allocator_type()):
        allocator_{alloc},
        size_{n},
        begin_{allocate(n)}
    {
        fill(begin_, begin_+size_, v);
    }
    //construct storage from host iterators range
    template<typename It, std::enable_if_t<detail::is_iterator<It>,int> =0 >
    cuda_aware_storage(It first, It last, const allocator_type& alloc = allocator_type()):
        allocator_{alloc},
        size_{std::distance(first,last)},
        begin_{allocate(size_)}
    {
        copy(first,last,begin_);
    }
    //construct storage from host init list
    cuda_aware_storage(std::initializer_list<value_type> init_data, const allocator_type& alloc = allocator_type()):
        allocator_{alloc},
        size_{static_cast<size_type>(init_data.size())},
        begin_{allocate(size_)}
    {
        copy(init_data.begin(),init_data.end(),begin_);
    }
    //construct storage from cuda aware pointers range, if storage allocates on device it use current active device of calling thread
    cuda_aware_storage(const_pointer first, const_pointer last, const allocator_type& alloc = allocator_type()):
        allocator_{alloc},
        size_{distance(first,last)},
        begin_{allocate(size_)}
    {
        copy(first,last,begin_);
    }

    auto data(){return begin_;}
    auto data()const{return  static_cast<const_pointer>(begin_);}
    auto begin(){return begin_;}
    auto end(){return  begin_ + size_;}
    auto begin()const{return static_cast<const_pointer>(begin_);}
    auto end()const{return  static_cast<const_pointer>(begin_+size_);}
    auto size()const{return size_;}
    auto empty()const{return !static_cast<bool>(begin_);}
    auto clone()const{return cuda_aware_storage{*this};}
    void free(){deallocate();}
    auto get_allocator()const{return allocator_;}

private:
    //private copy constructor, use clone() to make copy, if storage allocates on device it use current active device of calling thread
    cuda_aware_storage(const cuda_aware_storage& other):
        allocator_{std::allocator_traits<allocator_type>::select_on_container_copy_construction(other.get_allocator())},
        size_(other.size_),
        begin_(allocate(size_))
    {
        copy(other.begin(),other.end(),begin_);
    }

    //no copy assign other's allocator
    void copy_assign(const cuda_aware_storage& other, std::false_type){
        auto other_size = other.size();
        if (size()!=other_size){
            auto new_buffer = allocate(other_size);
            deallocate();
            size_ = other_size;
            begin_ = new_buffer;
        }
        copy(other.begin(),other.end(),begin_);
    }

    //copy assign other's allocator
    void copy_assign(const cuda_aware_storage& other, std::true_type){
        if (allocator_ ==  other.allocator_ || std::allocator_traits<allocator_type>::is_always_equal()){
            copy_assign(other, std::false_type{});
        }else{
            auto other_size = other.size();
            auto other_allocator = other.get_allocator();
            auto new_buffer = other_allocator.allocate(other_size);
            deallocate();
            size_ = other_size;
            begin_ = new_buffer;
            allocator_ = other_allocator;
        }
        copy(other.begin(),other.end(),begin_);
    }

    //no move assign other's allocator, if allocators not equal copy are made
    void move_assign(cuda_aware_storage&& other, std::false_type){
        if (allocator_ ==  other.allocator_ || std::allocator_traits<allocator_type>::is_always_equal()){
            deallocate();
            size_ = other.size_;
            begin_ = other.begin_;
            other.size_ = 0;
            other.begin_ = nullptr;
        }else{
            copy_assign(other, std::false_type{});
        }
    }

    //move assign other's allocator
    void move_assign(cuda_aware_storage&& other, std::true_type){
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