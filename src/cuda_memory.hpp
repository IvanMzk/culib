#ifndef CUDA_MEMORY_HPP_
#define CUDA_MEMORY_HPP_

#include <iterator>
#include <memory>
#include <immintrin.h>
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

//pointer to device memory
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

//pointer to page-locked host memory
template<typename T>
class locked_pointer : public basic_pointer<T,locked_pointer>
{
    static_assert(std::is_trivially_copyable_v<T>);
public:
    using iterator_category = std::random_access_iterator_tag;
    using typename basic_pointer::difference_type;
    using typename basic_pointer::value_type;
    using typename basic_pointer::pointer;
    using reference = T&;
    using const_reference = const T&;

    locked_pointer() = default;
    explicit locked_pointer(pointer p):
        basic_pointer{p}
    {}
    operator locked_pointer<const value_type>()const{return locked_pointer<const value_type>{get()};}
    using basic_pointer::operator=;
    auto& operator*()const{return *get();}
    auto& operator[](difference_type i)const{return *(*this+i);}
};

/*
* allocate memory on current active device
*/
template<typename T>
class device_allocator
{
    static_assert(std::is_trivially_copyable_v<T>);
public:
    using value_type = T;
    using pointer = device_pointer<T>;
    using const_pointer = device_pointer<const T>;
    using difference_type = typename pointer::difference_type;
    using size_type = difference_type;

    pointer allocate(size_type n){
        void* p{nullptr};
        if (n){
            cuda_error_check(cudaMalloc(&p,n*sizeof(T)));
        }
        return pointer{static_cast<T*>(p), cuda_get_device()};
    }
    void deallocate(pointer p, size_type){
        if (p){
            device_switcher switcher{p.device()};
            cuda_error_check(cudaFree(ptr_to_void(p)));
        }
    }
    bool operator==(const device_allocator& other)const{return true;}
};

/*
* allocate page-locked memory on host
*/
template<typename T>
class locked_allocator
{
    static_assert(std::is_trivially_copyable_v<T>);
    unsigned int flags;
public:
    using value_type = T;
    using pointer = locked_pointer<T>;
    using const_pointer = locked_pointer<const T>;
    using difference_type = typename pointer::difference_type;
    using size_type = difference_type;
    using is_aways_equal = std::true_type;

    explicit locked_allocator(unsigned int flags_ = cudaHostAllocDefault):
        flags{flags_}
    {}
    pointer allocate(size_type n){
        void* p{nullptr};
        if (n){
            cuda_error_check(cudaHostAlloc(&p,n*sizeof(T),flags));
        }
        return pointer{static_cast<T*>(p)};
    }
    void deallocate(pointer p, size_type){
        if (p){
            cuda_error_check(cudaFreeHost(ptr_to_void(p)));
        }
    }
    bool operator==(const locked_allocator& other)const{return flags == other.flags;}
};

/*
* allocate page-locked memory on host
*/
template<typename T>
class registered_allocator
{
    static_assert(std::is_trivially_copyable_v<T>);
    unsigned int flags;
public:
    using value_type = T;
    using pointer = locked_pointer<T>;
    using const_pointer = locked_pointer<const T>;
    using difference_type = typename pointer::difference_type;
    using size_type = difference_type;
    using is_aways_equal = std::true_type;

    explicit registered_allocator(unsigned int flags_ = cudaHostRegisterDefault):
        flags{flags_}
    {}
    pointer allocate(size_type n){
        unsigned char* p{nullptr};
        if (n){
            auto n_bytes = n*sizeof(value_type);
            p = new unsigned char[n_bytes];
            cuda_error_check(cudaHostRegister(p,n_bytes,flags));
        }
        return pointer{reinterpret_cast<T*>(p)};
    }
    void deallocate(pointer p, size_type){
        if (p){
            cuda_error_check(cudaHostUnregister(p.get()));
            delete[] reinterpret_cast<unsigned char*>(p.get());
        }
    }
    bool operator==(const registered_allocator& other)const{return flags == other.flags;}
};




/*
* allocate pageable memory on host
*/
template<typename T>
class pageable_allocator
{
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using difference_type = std::ptrdiff_t;
    using size_type = difference_type;

    pointer allocate(size_type n){
        void* p{nullptr};
        if (n){
            p = new char[n*sizeof(T)];
        }
        return static_cast<pointer>(p);
    }
    void deallocate(pointer p, size_type){
        if (p){
            delete[] reinterpret_cast<char*>(p);
        }
    }
    bool operator==(const pageable_allocator& other)const{return true;}
};

template<typename T, typename Alloc>
class memory_buffer
{
public:
    using allocator_type = Alloc;
    using value_type = typename allocator_type::value_type;
    using pointer = typename allocator_type::pointer;
    using size_type = typename allocator_type::size_type;
    memory_buffer(const memory_buffer&) = delete;
    memory_buffer& operator=(const memory_buffer&) = delete;
    memory_buffer& operator=(memory_buffer&&) = delete;
    ~memory_buffer(){deallocate();}
    memory_buffer(const size_type& n, const allocator_type& alloc = allocator_type{}):
        allocator_{alloc},
        size_{n},
        begin_{allocate(n)}
    {}
    memory_buffer(memory_buffer&& other):
        allocator_{std::move(other.allocator_)},
        size_{other.size_},
        begin_{other.begin_}
    {
        other.size_ = 0;
        other.begin_ = nullptr;
    }
    auto data()const{return begin_;}
    auto size()const{return size_;}
    auto begin()const{return begin_;}
    auto end()const{return begin_+size_;}
private:
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
template<typename T> using device_buffer = memory_buffer<T, device_allocator<T>>;
template<typename T> using locked_buffer = memory_buffer<T, locked_allocator<T>>;
template<typename T> using registered_buffer = memory_buffer<T, registered_allocator<T>>;
template<typename T> using pageable_buffer = memory_buffer<T, std::allocator<T>>;
using copy_buffer = locked_buffer<unsigned char>;


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

// template<typename>
// auto clz(){
//     static constexpr std::size_t hash[] = {0,1,3,7,14,29,27,22,13,26,21,11,23,15,31,30,28,25,19,6,12,24,17,2,5,10,20,9,18,4,8,16};
//     //0x76BE629;
//     //0x022fdd63cc95386d;
// }

inline auto alignment(void* p){
   return (reinterpret_cast<std::uintptr_t>(p) & (~reinterpret_cast<std::uintptr_t>(p) + 1));
}

template<std::size_t A>
auto align(void* p){
    static_assert(A != 0);
    static_assert((A&(A-1))  == 0);
    return reinterpret_cast<void *>((reinterpret_cast<std::uintptr_t>(p)+(A-1)) & ~(A-1));
}

// template<typename BlockT>
// auto align_for_copy(void* src, void* dst, std::size_t n_src, std::size_t n_dst){
//     static constexpr auto block_size = sizeof(BlockT);
//     static constexpr auto block_alignment = alignof(BlockT);
//     struct aligned_for_copy{
//         unsigned char* src_begin_aligned{reinterpret_cast<unsigned char*>(align<block_alignment>(src))};
//         std::ptrdiff_t src_begin_offset{src_begin_aligned - reinterpret_cast<unsigned char*>(src)};
//         unsigned char* dst_begin_aligned{reinterpret_cast<unsigned char*>(align<block_alignment>(dst))};
//         std::ptrdiff_t dst_begin_offset{dst_begin_aligned - reinterpret_cast<unsigned char*>(dst)};
//         std::size_t aligned_blocks_in_src{n_src>src_begin_offset ? (n_src - src_begin_offset)/block_size : std::size_t(0)};
//         std::size_t aligned_blocks_in_dst{n_dst>dst_begin_offset ? (n_dst - dst_begin_offset)/block_size : std::size_t(0)};
//         unsigned char* src_end_aligned{src_begin_aligned + aligned_blocks_in_src*block_size};
//         unsigned char* dst_end_aligned{dst_begin_aligned + aligned_blocks_in_dst*block_size};
//         unsigned char* src_end{reinterpret_cast<unsigned char*>(src)+n_src};
//         unsigned char* dst_end{reinterpret_cast<unsigned char*>(dst)+n_src};
//     };
//     return aligned_for_copy{};
// }

template<typename BlockT>
class aligned_for_copy{
public:
    aligned_for_copy(void* first__, std::size_t n__):
        first_{first__},
        n_{n__}
    {}
    constexpr auto block_size()const{return block_size_;}
    constexpr auto block_alignment()const{return block_alignment_;}
    auto first()const{return first_;}
    auto n()const{return n_;}
    auto first_aligned()const{return first_aligned_;}
    auto first_offset()const{return first_offset_;}
    auto aligned_blocks()const{return aligned_blocks_;}
    auto last()const{return last_;}
    auto last_aligned()const{return last_aligned_;}
    auto last_offset()const{return last_offset_;}
private:
    static constexpr auto block_size_ = sizeof(BlockT);
    static constexpr auto block_alignment_ = alignof(BlockT);
    void* first_;
    std::size_t n_;
    void* first_aligned_{align<block_alignment_>(first_)};
    std::ptrdiff_t first_offset_{reinterpret_cast<unsigned char*>(first_aligned_) - reinterpret_cast<unsigned char*>(first_)};
    std::size_t aligned_blocks_{n_>first_offset_ ? (n_ - first_offset_)/block_size_ : std::size_t{0}};

    void* last_{reinterpret_cast<unsigned char*>(first_)+n_};
    void* last_aligned_{reinterpret_cast<unsigned char*>(first_aligned_) + aligned_blocks_*block_size_};
    std::size_t last_offset_{reinterpret_cast<unsigned char*>(last_) > reinterpret_cast<unsigned char*>(last_aligned_) ?
        reinterpret_cast<unsigned char*>(last_) - reinterpret_cast<unsigned char*>(last_aligned_) : std::size_t{0}};
};

//copy routines to transfer between host and device, parameters of ordinary pointers types treats as pointers to pageable host memory
//copy from host to device
// template<typename T>
// void copy(const T* first, const T* last, device_pointer<T> d_first){
//     auto n = std::distance(first,last);
//     cuda_error_check(cudaMemcpyAsync(ptr_to_void(d_first), ptr_to_void(first), n*sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice, cuda_stream{}));
// }

template<typename T>
void copy_locked(locked_pointer<T> first, locked_pointer<T> last, device_pointer<std::remove_const_t<T>> d_first, cudaStream_t stream = cuda_stream{}){
    auto n = std::distance(first,last);
    //std::cout<<std::endl<<"void copy(locked_pointer<T> first, locked_pointer<T> last, device_pointer<std::remove_const_t<T>> d_first, cudaStream_t stream = cuda_stream{}){"<<n;
    cuda_error_check(cudaMemcpyAsync(ptr_to_void(d_first), ptr_to_void(first), n*sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice, stream));
}

template<typename It, std::enable_if_t<!is_basic_pointer_v<It>,int> =0 >
void copy_pageable(It first, It last, device_pointer<typename std::iterator_traits<It>::value_type> d_first){
    auto n = std::distance(first,last);
    //std::cout<<std::endl<<"void copy(It first, It last, device_pointer<typename std::iterator_traits<It>::value_type> d_first){"<<n;
    auto buffer = locked_buffer<typename std::iterator_traits<It>::value_type>(n);
    std::uninitialized_copy_n(first,n,buffer.data().get());
    copy_locked(buffer.begin(),buffer.end(),d_first);
}

//copy from device to host
// template<typename T>
// void copy(device_pointer<T> first, device_pointer<T> last, std::remove_const_t<T>* d_first){
//     auto n = distance(first,last);
//     cuda_error_check(cudaMemcpyAsync(ptr_to_void(d_first), ptr_to_void(first), n*sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost, cuda_stream{}));
// }
// template<typename T>
// void copy(device_pointer<T> first, device_pointer<T> last, locked_pointer<std::remove_const_t<T>> d_first, cudaStream_t stream = cuda_stream{}){
//     auto n = distance(first,last);
//     cuda_error_check(cudaMemcpyAsync(ptr_to_void(d_first), ptr_to_void(first), n*sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost, stream));
// }

// template<typename T, typename It, std::enable_if_t<!is_basic_pointer_v<It>,int> =0>
// void copy(device_pointer<T> first, device_pointer<T> last, It d_first){
//     static_assert(std::is_same_v<std::decay_t<T>, typename std::iterator_traits<It>::value_type>);
//     auto n = distance(first,last);
//     auto buffer = locked_buffer<std::iterator_traits<It>::value_type>(n);
//     copy(first,last,buffer.begin());
//     std::copy_n(buffer.data().get(),n,d_first);
// }

//copy from device to device, src and dst must be allocated on same device
// template<typename T>
// void copy(device_pointer<T> first, device_pointer<T> last, device_pointer<std::remove_const_t<T>> d_first){
//     auto n = distance(first,last);
//     cuda_error_check(cudaMemcpyAsync(ptr_to_void(d_first), ptr_to_void(first), n*sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToDevice, cuda_stream{}));
// }

//fill device memory in range first last with v
template<typename T>
void fill(device_pointer<T> first, device_pointer<T> last, const T& v){
    auto n = distance(first,last);
    auto buffer = locked_buffer<T>(n);
    std::uninitialized_fill_n(buffer.data().get(),n,v);
    copy(buffer.begin(), buffer.end(), first);
}
// template<typename T>
// void fill(device_pointer<T> first, device_pointer<T> last, const T& v){
//     auto n = distance(first,last);
//     auto buffer = make_locked_memory_buffer<T>(n, cudaHostAllocWriteCombined);
//     std::uninitialized_fill_n(buffer.get(),n,v);
//     copy(buffer.get(), buffer.get()+n, first);
// }

}   //end of namespace cuda_experimental

#endif