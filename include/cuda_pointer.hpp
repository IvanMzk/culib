/*
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#ifndef CUDA_POINTER_HPP_
#define CUDA_POINTER_HPP_

#include <type_traits>
#include <iterator>
#include <cstdint>

namespace culib{

template<typename T, template<typename> typename D> class basic_pointer;

namespace detail{

template<typename T>
struct is_basic_pointer{
    template<typename U, template<typename> typename V>
    static std::true_type selector(const basic_pointer<U,V>&);
    static std::false_type selector(...);
    using type = decltype(selector(std::declval<T>()));
};
template<typename T> inline constexpr bool is_basic_pointer_v = typename is_basic_pointer<T>::type();

}   //end of namespace detail

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
    template<typename U, std::enable_if_t<!detail::is_basic_pointer_v<U>,int> =0>
    friend auto operator+(const basic_pointer& lhs, const U& rhs){
        derived_type res{static_cast<const derived_type&>(lhs)};
        res.set_ptr(lhs.get() + rhs);
        return res;
    }
    operator bool()const{return static_cast<bool>(ptr);}
    operator T*()const{return ptr;}
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

template<typename T, template<typename> typename D, typename U, std::enable_if_t<!detail::is_basic_pointer_v<U>,int> =0>
auto operator+(const U& lhs, const basic_pointer<T,D>& rhs){return rhs+lhs;}
template<typename T, template<typename> typename D, typename U, std::enable_if_t<!detail::is_basic_pointer_v<U>,int> =0 >
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
//return aligned pointer that is nearest to p and greater or equal to p
//A - required alignment in bytes
template<std::size_t A>
inline auto align(const void* p){
    static_assert(A != 0);
    static_assert((A&(A-1))  == 0);
    return reinterpret_cast<const void *>((reinterpret_cast<std::uintptr_t>(p)+(A-1)) & ~(A-1));
}
template<std::size_t A>
inline auto align(void* p){
    return const_cast<void*>(align<A>(const_cast<const void*>(p)));
}
inline auto alignment(const void* p){
   return (reinterpret_cast<std::uintptr_t>(p) & (~reinterpret_cast<std::uintptr_t>(p) + 1));
}

//pointer to device memory
template<typename T>
class device_pointer : public basic_pointer<T,device_pointer>
{
    static_assert(std::is_trivially_copyable_v<T>);
    using basic_poiner_base = basic_pointer<T,device_pointer>;
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
    using typename basic_poiner_base::difference_type;
    using typename basic_poiner_base::value_type;
    using typename basic_poiner_base::pointer;
    using reference = std::conditional_t<std::is_const_v<T>,T, device_data_reference>;
    using const_reference = std::conditional_t<std::is_const_v<T>,T, device_data_reference>;
    using device_id_type = int;
    static constexpr device_id_type undefined_device = -1;
    device_pointer(std::nullptr_t):
        basic_poiner_base{nullptr},
        device_{undefined_device}
    {}
    device_pointer():
        device_pointer{nullptr}
    {}
    device_pointer(pointer p, device_id_type device__):
        basic_poiner_base{p},
        device_{device__}
    {}
    using basic_poiner_base::get;
    using basic_poiner_base::operator=;
    explicit operator device_pointer<const value_type>()const{return device_pointer<const value_type>{get(),device()};}
    auto operator*()const{return deref_helper(typename std::is_const<T>::type{});}
    auto operator[](difference_type i)const{return *(*this+i);}
    auto device()const{return device_;}
};

template<typename T>
auto operator==(const device_pointer<T>& lhs, const device_pointer<T>& rhs){return lhs - rhs == typename device_pointer<T>::difference_type(0) && lhs.device() == rhs.device();}
template<typename T>
auto operator!=(const device_pointer<T>& lhs, const device_pointer<T>& rhs){return !(lhs == rhs);}

//pointer to page-locked host memory
template<typename T>
class locked_pointer : public basic_pointer<T,locked_pointer>
{
    static_assert(std::is_trivially_copyable_v<T>);
    using basic_poiner_base = basic_pointer<T,locked_pointer>;
public:
    using iterator_category = std::random_access_iterator_tag;
    using typename basic_poiner_base::difference_type;
    using typename basic_poiner_base::value_type;
    using typename basic_poiner_base::pointer;
    using reference = T&;
    using const_reference = const T&;

    locked_pointer():
        basic_poiner_base{nullptr}
    {}
    explicit locked_pointer(pointer p):
        basic_poiner_base{p}
    {}
    using basic_poiner_base::operator=;
    using basic_poiner_base::get;
    explicit operator locked_pointer<const value_type>()const{return locked_pointer<const value_type>{get()};}
    auto& operator*()const{return *get();}
    auto& operator[](difference_type i)const{return *(*this+i);}
};


}   //end of namespace culib

#endif