#ifndef DEVICE_STORAGE_HPP_
#define DEVICE_STORAGE_HPP_

#include <memory>
#include "cuda_memory.hpp"

namespace cuda_experimental{

namespace detail{
    template<typename, typename = void> constexpr bool is_iterator = false;
    template<typename T> constexpr bool is_iterator<T,std::void_t<typename std::iterator_traits<T>::iterator_category>> = true;
}   //end of namespace detail


template<typename T, typename MemoryHandler = cuda_memory_handler<T>>
class device_storage : private MemoryHandler
{
    using memory_handler_type = MemoryHandler;
public:
    using difference_type = typename memory_handler_type::difference_type;
    using size_type = typename memory_handler_type::size_type;
    using value_type = typename memory_handler_type::value_type;
    using pointer = typename memory_handler_type::pointer;

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
        auto buffer = std::make_unique<value_type[]>(n);
        std::uninitialized_fill_n(buffer.get(),n,v);
        memcpy_host_to_device(begin_,buffer.get(),n);
    }
    //construct storage from host iterators range
    template<typename It, std::enable_if_t<detail::is_iterator<It> ,int> =0 >
    device_storage(It first, It last):
        size_{std::distance(first,last)},
        begin_{allocate(size_)}
    {
        auto buffer = std::make_unique<value_type[]>(size_);
        std::uninitialized_copy(first,last,buffer.get());
        memcpy_host_to_device(begin_,buffer.get(),size_);
    }
    //construct storage from host init list
    device_storage(std::initializer_list<value_type> init_data):
        size_{static_cast<size_type>(init_data.size())},
        begin_{allocate(size_)}
    {
        memcpy_host_to_device(begin_,init_data.begin(),size_);
    }
    //construct storage from device iterators range
    device_storage(pointer first, pointer last):
        size_{distance(first,last)},
        begin_{allocate(size_)}
    {
        memcpy_device_to_device(begin_,first,size_);
    }

    auto device_begin()const{return begin_;}
    auto device_end()const{return  begin_ + size_;}
    auto size()const{return size_;}
    auto clone()const{return device_storage{*this};}
    void free(){deallocate();}

private:
    device_storage(const device_storage& other):
        size_{other.size_},
        begin_{allocate(other.size_)}
    {
        memcpy_device_to_device(begin_,other.begin_,size_);
    }

    pointer allocate(const size_type& n){
        return memory_handler_type::allocate(n);
    }
    void deallocate(){
        if (begin_){
            memory_handler_type::deallocate(begin_,size_);
            size_ = 0;
            begin_ = nullptr;
        }
    }

    size_type size_;
    pointer begin_;
};

}   //end of namespace cuda_experimental



#endif