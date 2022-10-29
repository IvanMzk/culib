#ifndef CUDA_MEMORY_HPP_
#define CUDA_MEMORY_HPP_

#include <iterator>
#include <memory>
#include "cuda_helpers.hpp"

namespace cuda_experimental{

template<typename T, template<typename> typename D> class basic_pointer;

template<typename T>
class is_basic_pointer_t{
    template<typename...U>
    static std::true_type selector(const basic_pointer<U...>&);
    static std::false_type selector(...);
public: using type = decltype(selector(std::declval<T>()));
};
template<typename T> constexpr bool is_basic_pointer_v = is_basic_pointer_t<T>::type();

template<typename T, template<typename> typename D>
class basic_pointer{
    using derived_type = D<T>;
public:
    using value_type = T;
    using pointer = T*;
    using difference_type = std::ptrdiff_t;

    basic_pointer& operator=(const basic_pointer&) = default;
    basic_pointer& operator=(basic_pointer&&) = default;
    derived_type& operator=(std::nullptr_t){
        ptr = nullptr;
        return to_derived();
    }
    derived_type& operator++(){
        ++ptr;
        return to_derived();
    }
    derived_type& operator--(){
        --ptr;
        return to_derived();
    }
    template<typename U>
    derived_type& operator+=(const U& offset){
        ptr+=offset;
        return to_derived();
    }
    template<typename U>
    derived_type& operator-=(const U& offset){
        ptr-=offset;
        return to_derived();
    }
    template<typename U, std::enable_if_t<!is_basic_pointer_v<U>,int> =0>
    friend auto operator+(const basic_pointer& lhs, const U& rhs){
        derived_type res{static_cast<const derived_type&>(lhs)};
        res.set_ptr(lhs.get() + rhs);
        return res;
    }
    operator bool()const{return static_cast<bool>(ptr);}
    pointer get()const{return ptr;}
private:
    friend derived_type;
    basic_pointer(const basic_pointer&) = default;
    basic_pointer(basic_pointer&&) = default;
    explicit basic_pointer(pointer ptr_ = nullptr):
        ptr{ptr_}
    {}
    auto& to_derived(){return static_cast<derived_type&>(*this);}
    void set_ptr(pointer ptr_){ptr = ptr_;}
    pointer ptr;
};

template<typename T, template<typename> typename D>
auto operator++(basic_pointer<T,D>& lhs, int){
    D<T> res{static_cast<D<T>&>(lhs)};
    ++lhs;
    return res;
}
template<typename T, template<typename> typename D>
auto operator--(basic_pointer<T,D>& lhs, int){
    D<T> res{static_cast<D<T>&>(lhs)};
    --lhs;
    return res;
}

template<typename T, template<typename> typename D, typename U, std::enable_if_t<!is_basic_pointer_v<U>,int> =0>
auto operator+(const U& lhs, const basic_pointer<T,D>& rhs){return rhs+lhs;}
template<typename T, template<typename> typename D, typename U, std::enable_if_t<!is_basic_pointer_v<U>,int> =0 >
auto operator-(const basic_pointer<T,D>& lhs, const U& rhs){return lhs+-rhs;}

template<typename T, template<typename> typename D>
auto operator-(const basic_pointer<T,D>& lhs, const basic_pointer<T,D>& rhs){return lhs.get() - rhs.get();}
template<typename T, template<typename> typename D>
auto operator==(const basic_pointer<T,D>& lhs, const basic_pointer<T,D>& rhs){return lhs - rhs == typename basic_pointer<T,D>::difference_type(0);}
template<typename T, template<typename> typename D>
auto operator!=(const basic_pointer<T,D>& lhs, const basic_pointer<T,D>& rhs){return !(lhs == rhs);}
template<typename T, template<typename> typename D>
auto operator>(const basic_pointer<T,D>& lhs, const basic_pointer<T,D>& rhs){return lhs - rhs > typename basic_pointer<T,D>::difference_type(0);}
template<typename T, template<typename> typename D>
auto operator<(const basic_pointer<T,D>& lhs, const basic_pointer<T,D>& rhs){return rhs - lhs > typename basic_pointer<T,D>::difference_type(0);}
template<typename T, template<typename> typename D>
auto operator>=(const basic_pointer<T,D>& lhs, const basic_pointer<T,D>& rhs){return !(lhs < rhs);}
template<typename T, template<typename> typename D>
auto operator<=(const basic_pointer<T,D>& lhs, const basic_pointer<T,D>& rhs){return !(lhs > rhs);}

template<typename T, template<typename> typename D>
auto distance(const basic_pointer<T,D>& begin, const basic_pointer<T,D>& end){return end-begin;}
template<typename T, template<typename> typename D>
auto ptr_to_void(const basic_pointer<T,D>& p){return static_cast<std::conditional_t<std::is_const_v<T>,const void*,void*>>(p.get());}
template<typename T>
auto ptr_to_void(const T* p){return static_cast<const void*>(p);}
template<typename T>
auto ptr_to_void(T* p){return static_cast<void*>(p);}
template<typename T, template<typename> typename D>
auto ptr_to_const(const basic_pointer<T,D>& p){return static_cast<D<const T>>(static_cast<const D<T>&>(p));}

template<typename...T>
inline auto cuda_pointer_get_attributes(const basic_pointer<T...>& p){
    cudaPointerAttributes attr;
    cudaPointerGetAttributes(&attr, p.get());
    return attr;
}

template<typename T>
class device_pointer : public basic_pointer<T,device_pointer>
{
    static_assert(std::is_trivially_copyable_v<T>);
    class device_data_reference{
        device_pointer data;
    public:
        device_data_reference(device_pointer data_):
            data{data_}
        {}
        operator T()const{
            std::remove_const_t<T> buffer;
            copy(data, data+1, &buffer);
            return buffer;
        }
        T operator=(const T& v){
            copy(&v, &v+1, data);
            return v;
        }
    };

    auto deref_helper(std::true_type)const{
        return static_cast<T>(device_data_reference{*this});
    }
    auto deref_helper(std::false_type)const{
        return device_data_reference{*this};
    }

    int device_;

public:
    using iterator_category = std::random_access_iterator_tag;
    using typename basic_pointer::difference_type;
    using typename basic_pointer::value_type;
    using typename basic_pointer::pointer;
    using reference = std::conditional_t<std::is_const_v<T>,T, device_data_reference>;
    using const_reference = std::conditional_t<std::is_const_v<T>,T, device_data_reference>;
    using device_id_type = int;
    static constexpr device_id_type undefined_device = -1;
    device_pointer():
        basic_pointer{nullptr},
        device_{undefined_device}
    {}
    device_pointer(pointer p, device_id_type device__):
        basic_pointer{p},
        device_{device__}
    {}
    operator device_pointer<const value_type>()const{return device_pointer<const value_type>{get(),device()};}
    using basic_pointer::operator=;
    auto operator*()const{return deref_helper(std::is_const<T>::type{});}
    auto operator[](difference_type i)const{return *(*this+i);}
    auto device()const{return device_;}
};

/*
* allocate device memory on current active device
* deallocate device memory on device it has been allocated
*/
template<typename T>
class device_allocator
{
public:
    using value_type = T;
    using pointer = device_pointer<T>;
    using const_pointer = device_pointer<const T>;
    using difference_type = typename pointer::difference_type;
    using size_type = difference_type;

    pointer allocate(size_type n){
        void* p;
        cuda_error_check(cudaMalloc(&p,n*sizeof(T)));
        return pointer{static_cast<T*>(p), cuda_get_device()};
    }
    void deallocate(pointer p, size_type){
        device_switcher switcher{p.device()};
        cuda_error_check(cudaFree(ptr_to_void(p)));
    }
    bool operator==(const device_allocator& other)const{return true;}
};

template<typename T, typename SizeT>
auto make_locked_memory_buffer(const SizeT& n, unsigned int flags = cudaHostAllocDefault){
    void* p;
    cuda_error_check(cudaHostAlloc(&p,n*sizeof(T),flags));
    auto deleter = [](T* p_){cudaFreeHost(p_);};
    return std::unique_ptr<T,decltype(deleter)>(static_cast<T*>(p), deleter);
}

template<typename T, typename SizeT>
auto make_pageable_memory_buffer(const SizeT& n){
    return std::make_unique<T[]>(n);
}

//copy routines to transfer between host and device, parameters of ordinary pointers types treats as pointers to host memory
//copy from host to device
template<typename T>
void copy(const T* first, const T* last, device_pointer<T> d_first){
    auto n = std::distance(first,last);
    cuda_error_check(cudaMemcpy(ptr_to_void(d_first), ptr_to_void(first), n*sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice));
}

template<typename It, std::enable_if_t<!std::is_pointer_v<It> && !is_basic_pointer_v<It>,int> =0 >
void copy(It first, It last, device_pointer<typename std::iterator_traits<It>::value_type> d_first){
    static_assert(!std::is_pointer_v<It>);
    auto n = std::distance(first,last);
    auto buffer = make_locked_memory_buffer<std::iterator_traits<It>::value_type>(n,cudaHostAllocWriteCombined);
    std::uninitialized_copy_n(first,n,buffer.get());
    copy(buffer.get(),buffer.get()+n,d_first);
}

//copy from device to host
template<typename T>
void copy(device_pointer<T> first, device_pointer<T> last, std::remove_const_t<T>* d_first){
    auto n = distance(first,last);
    cuda_error_check(cudaMemcpy(ptr_to_void(d_first), ptr_to_void(first), n*sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost));
}

template<typename T, typename It, std::enable_if_t<!std::is_pointer_v<It> && !is_basic_pointer_v<It>,int> =0>
void copy(device_pointer<T> first, device_pointer<T> last, It d_first){
    static_assert(!std::is_pointer_v<It>);
    static_assert(std::is_same_v<std::decay_t<T>, typename std::iterator_traits<It>::value_type>);
    auto n = distance(first,last);
    auto buffer = make_locked_memory_buffer<std::iterator_traits<It>::value_type>(n);
    copy(first,last,buffer.get());
    std::copy_n(buffer.get(),n,d_first);
}

//copy from device to device, src and dst must be allocated on same device
template<typename T>
void copy(device_pointer<T> first, device_pointer<T> last, device_pointer<std::remove_const_t<T>> d_first){
    auto n = distance(first,last);
    cuda_error_check(cudaMemcpy(ptr_to_void(d_first), ptr_to_void(first), n*sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToDevice));
}

//fill device memory in range first last with v
template<typename T>
void fill(device_pointer<T> first, device_pointer<T> last, const T& v){
    auto n = distance(first,last);
    auto buffer = make_locked_memory_buffer<T>(n, cudaHostAllocWriteCombined);
    std::uninitialized_fill_n(buffer.get(),n,v);
    copy(buffer.get(), buffer.get()+n, first);
}

}   //end of namespace cuda_experimental

#endif