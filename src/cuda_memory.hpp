#ifndef CUDA_MEMORY_HPP_
#define CUDA_MEMORY_HPP_

#include <iterator>
#include <memory>
#include "cuda_helpers.hpp"

namespace cuda_experimental{

template<typename T, template<typename> typename D>
class basic_pointer{
    using derived_type = D<T>;
    static_assert(std::is_trivially_copyable_v<T>);
public:
    using value_type = T;
    using pointer = T*;
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
    operator bool()const{return static_cast<bool>(ptr);}
    operator D<const T>()const{return D<const T>{ptr};}
    pointer get()const{return ptr;}
private:
    friend derived_type;
    basic_pointer(const basic_pointer&) = default;
    basic_pointer(basic_pointer&&) = default;
    explicit basic_pointer(pointer ptr_ = nullptr):
        ptr{ptr_}
    {}
    auto& to_derived(){return static_cast<derived_type&>(*this);}
    pointer ptr;
};

template<typename T>
class is_basic_pointer_t{
    template<typename...U>
    static std::true_type selector(const basic_pointer<U...>&);
    static std::false_type selector(...);
public: using type = decltype(selector(std::declval<T>()));
};
template<typename T> constexpr bool is_basic_pointer_v = is_basic_pointer_t<T>::type();

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
auto operator+(const basic_pointer<T,D>& lhs, const U& rhs){return D<T>{lhs.get() + rhs};}
template<typename T, template<typename> typename D, typename U, std::enable_if_t<!is_basic_pointer_v<U>,int> =0>
auto operator+(const U& lhs, const basic_pointer<T,D>& rhs){return rhs+lhs;}
template<typename T, template<typename> typename D, typename U, std::enable_if_t<!is_basic_pointer_v<U>,int> =0 >
auto operator-(const basic_pointer<T,D>& lhs, const U& rhs){return lhs+-rhs;}

template<typename T, template<typename> typename D>
auto operator==(const basic_pointer<T,D>& lhs, const basic_pointer<T,D>& rhs){return lhs.get() == rhs.get();}
template<typename T, template<typename> typename D>
auto operator!=(const basic_pointer<T,D>& lhs, const basic_pointer<T,D>& rhs){return !(lhs == rhs);}
template<typename T, template<typename> typename D>
auto operator-(const basic_pointer<T,D>& lhs, const basic_pointer<T,D>& rhs){return lhs.get() - rhs.get();}
template<typename T, template<typename> typename D>
auto distance(const basic_pointer<T,D>& begin, const basic_pointer<T,D>& end){return end-begin;}
template<typename T, template<typename> typename D>
auto ptr_to_void(const basic_pointer<T,D>& p){return static_cast<std::conditional_t<std::is_const_v<T>,const void*,void*>>(p.get());}
template<typename T>
auto ptr_to_void(const T* p){return static_cast<const void*>(p);}
template<typename T>
auto ptr_to_void(T* p){return static_cast<void*>(p);}
template<typename T, template<typename> typename D>
auto ptr_to_const(const basic_pointer<T,D>& p){return static_cast<D<const T>>(p);}

template<typename T>
class device_pointer : public basic_pointer<T,device_pointer>
{
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

    auto deref_helper(std::true_type){
        return static_cast<T>(device_data_reference{*this});
    }
    auto deref_helper(std::false_type){
        return device_data_reference{*this};
    }

public:
    using typename basic_pointer::value_type;
    using typename basic_pointer::pointer;
    device_pointer(pointer p= nullptr):
        basic_pointer{p}
    {}
    using basic_pointer::operator=;
    auto operator*(){return deref_helper(std::is_const<T>::type{});}
};


template<typename T>
class device_allocator
{
public:
    using difference_type = std::ptrdiff_t;
    using size_type = difference_type;
    using value_type = T;
    using pointer = device_pointer<T>;
    using const_pointer = device_pointer<const T>;

    pointer allocate(size_type n){
        void* p;
        cuda_error_check(cudaMalloc(&p,n*sizeof(T)));
        return pointer{static_cast<T*>(p)};
    }
    void deallocate(pointer p, size_type){
        cuda_error_check(cudaFree(ptr_to_void(p)));
    }
    bool operator==(const device_allocator& other)const{return true;}
};


// template<typename T>
// class cuda_allocator
// {
//     using device_id_type = int;
// public:
//     using difference_type = std::ptrdiff_t;
//     using size_type = difference_type;
//     using value_type = T;
//     using pointer = basic_pointer<T>;
//     using const_pointer = basic_pointer<const T>;
//     using propagate_on_container_copy_assignment = std::true_type;
//     using propagate_on_container_move_assignment = std::true_type;

//     cuda_allocator(const cuda_allocator&) = default;
//     cuda_allocator(cuda_allocator&&) = default;
//     cuda_allocator& operator=(const cuda_allocator&) = default;
//     cuda_allocator& operator=(cuda_allocator&&) = default;
//     cuda_allocator(device_id_type managed_device_id_ = cuda_get_device()):
//         managed_device_id{managed_device_id_}
//     {}
//     pointer allocate(size_type n){
//         return allocator_helper([this](size_type n){void* p; cuda_error_check(cudaMalloc(&p,n*sizeof(T))); return pointer{static_cast<T*>(p)};}, n);
//     }
//     void deallocate(pointer p, size_type){
//         return allocator_helper([this](pointer p){cuda_error_check(cudaFree(ptr_to_void(p)));}, p);
//     }
//     bool operator==(const cuda_allocator& other)const{return managed_device_id == other.managed_device_id;}
// private:
//     struct device_restorer{
//         ~device_restorer(){cuda_error_check(cudaSetDevice(device));}
//         device_id_type device{cuda_get_device()};
//     };
//     template<typename F, typename...Args>
//     auto allocator_helper(const F& operation, Args&&...args){
//         device_restorer restorer{};
//         cuda_error_check(cudaSetDevice(managed_device_id));
//         return operation(std::forward<Args>(args)...);
//     }

//     device_id_type managed_device_id;
// };

// template<typename T>
// class unified_memory_allocator
// {
// public:
//     using difference_type = std::ptrdiff_t;
//     using size_type = difference_type;
//     using value_type = T;
//     using pointer = basic_pointer<T>;
//     using const_pointer = basic_pointer<const T>;

//     pointer allocate(size_type n){
//         void* p;
//         cuda_error_check(cudaMallocManaged(&p,n*sizeof(T)));
//         return pointer{static_cast<T*>(p)};
//     }
//     void deallocate(pointer p, size_type){
//         cuda_error_check(cudaFree(ptr_to_void(p)));
//     }
//     bool operator==(const unified_memory_allocator& other)const{return true;}
// };

// /*
// * host memory allocator for cuda unified addressing
// * allocate host memory or use already allocated host memory
// */
// template<typename T>
// class cuda_mapping_allocator
// {
// public:
//     using difference_type = std::ptrdiff_t;
//     using size_type = difference_type;
//     using value_type = T;
//     using pointer = basic_pointer<T>;
//     using const_pointer = basic_pointer<const T>;
//     using propagate_on_container_copy_assignment = std::true_type;
//     using propagate_on_container_move_assignment = std::true_type;

//     cuda_mapping_allocator(const cuda_mapping_allocator&) = default;
//     cuda_mapping_allocator(cuda_mapping_allocator&&) = default;
//     cuda_mapping_allocator& operator=(const cuda_mapping_allocator&) = default;
//     cuda_mapping_allocator& operator=(cuda_mapping_allocator&&) = default;
//     cuda_mapping_allocator(T* host_data_ = nullptr):
//         host_data{host_data_}
//     {}

//     pointer allocate(size_type n){
//         void* host_buffer;
//         if (host_data){
//             host_buffer = host_data;
//             //cudaHostRegisterDefault: On a system with unified virtual addressing, the memory will be both mapped and portable.
//             //On a system with no unified virtual addressing, the memory will be neither mapped nor portable.
//             cuda_error_check(cudaHostRegister(host_buffer,n*sizeof(T),cudaHostRegisterDefault));
//         }else{
//             host_buffer = make_host_locked_buffer<value_type>(n).release();
//         }
//         void* p;
//         cuda_error_check(cudaHostGetDevicePointer(&p, host_buffer, 0));
//         return pointer{static_cast<T*>(p)};
//     }
//     void deallocate(pointer p, size_type){
//         if (host_data){
//             cuda_error_check(cudaHostUnregister(host_data));
//         }else{
//             cuda_error_check(cudaFreeHost(ptr_to_void(p)));
//         }
//     }
//     bool operator==(const cuda_mapping_allocator& other)const{return host_data == other.host_data;}
// private:
//     T* host_data;
// };

template<typename T, typename SizeT>
auto make_host_locked_buffer(const SizeT& n, unsigned int flags = cudaHostAllocDefault){
    void* p;
    cuda_error_check(cudaHostAlloc(&p,n*sizeof(T),flags));
    auto deleter = [](T* p_){cudaFreeHost(p_);};
    return std::unique_ptr<T,decltype(deleter)>(static_cast<T*>(p), deleter);
}

template<typename T, typename SizeT>
auto make_host_buffer(const SizeT& n){
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
    auto buffer = make_host_locked_buffer<std::iterator_traits<It>::value_type>(n,cudaHostAllocWriteCombined);
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
    auto buffer = make_host_locked_buffer<std::iterator_traits<It>::value_type>(n);
    copy(first,last,buffer.get());
    std::copy_n(buffer.get(),n,d_first);
}
//copy from device to device, src and dst must be allocated on same device
template<typename T>
void copy(device_pointer<T> first, device_pointer<T> last, device_pointer<std::remove_const_t<T>> d_first){
    auto n = distance(first,last);
    cuda_error_check(cudaMemcpy(ptr_to_void(d_first), ptr_to_void(first), n*sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToDevice));
}

}   //end of namespace cuda_experimental

#endif