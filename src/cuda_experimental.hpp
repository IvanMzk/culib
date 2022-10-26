#ifndef CUDA_EXPERIMENTAL_HPP_
#define CUDA_EXPERIMENTAL_HPP_

#include "cuda_memory.hpp"

namespace cuda_experimental{

template<typename T>
class test_pointer : public basic_pointer<T,test_pointer>
{
    static_assert(std::is_trivially_copyable_v<T>);

public:
    using typename basic_pointer::difference_type;
    using typename basic_pointer::value_type;
    using typename basic_pointer::pointer;
    test_pointer(pointer p= nullptr):
        basic_pointer{p}
    {}
    using basic_pointer::operator=;
};


template<typename T>
class cuda_allocator
{
    using device_id_type = int;
public:
    using difference_type = std::ptrdiff_t;
    using size_type = difference_type;
    using value_type = T;
    using pointer = test_pointer<T>;
    using const_pointer = test_pointer<const T>;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;

    cuda_allocator(const cuda_allocator&) = default;
    cuda_allocator(cuda_allocator&&) = default;
    cuda_allocator& operator=(const cuda_allocator&) = default;
    cuda_allocator& operator=(cuda_allocator&&) = default;
    cuda_allocator(device_id_type managed_device_id_ = cuda_get_device()):
        managed_device_id{managed_device_id_}
    {}
    pointer allocate(size_type n){
        return allocator_helper([this](size_type n){void* p; cuda_error_check(cudaMalloc(&p,n*sizeof(T))); return pointer{static_cast<T*>(p)};}, n);
    }
    void deallocate(pointer p, size_type){
        return allocator_helper([this](pointer p){cuda_error_check(cudaFree(ptr_to_void(p)));}, p);
    }
    bool operator==(const cuda_allocator& other)const{return managed_device_id == other.managed_device_id;}
private:
    struct device_restorer{
        ~device_restorer(){cuda_error_check(cudaSetDevice(device));}
        device_id_type device{cuda_get_device()};
    };
    template<typename F, typename...Args>
    auto allocator_helper(const F& operation, Args&&...args){
        device_restorer restorer{};
        cuda_error_check(cudaSetDevice(managed_device_id));
        return operation(std::forward<Args>(args)...);
    }

    device_id_type managed_device_id;
};

template<typename T>
class unified_memory_allocator
{
public:
    using difference_type = std::ptrdiff_t;
    using size_type = difference_type;
    using value_type = T;
    using pointer = test_pointer<T>;
    using const_pointer = test_pointer<const T>;

    pointer allocate(size_type n){
        void* p;
        cuda_error_check(cudaMallocManaged(&p,n*sizeof(T)));
        return pointer{static_cast<T*>(p)};
    }
    void deallocate(pointer p, size_type){
        cuda_error_check(cudaFree(ptr_to_void(p)));
    }
    bool operator==(const unified_memory_allocator& other)const{return true;}
};

/*
* host memory allocator for cuda unified addressing
* allocate host memory or use already allocated host memory
*/
template<typename T>
class cuda_mapping_allocator
{
public:
    using difference_type = std::ptrdiff_t;
    using size_type = difference_type;
    using value_type = T;
    using pointer = test_pointer<T>;
    using const_pointer = test_pointer<const T>;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;

    cuda_mapping_allocator(const cuda_mapping_allocator&) = default;
    cuda_mapping_allocator(cuda_mapping_allocator&&) = default;
    cuda_mapping_allocator& operator=(const cuda_mapping_allocator&) = default;
    cuda_mapping_allocator& operator=(cuda_mapping_allocator&&) = default;
    cuda_mapping_allocator(T* host_data_ = nullptr):
        host_data{host_data_}
    {}

    pointer allocate(size_type n){
        void* host_buffer;
        if (host_data){
            host_buffer = host_data;
            //cudaHostRegisterDefault: On a system with unified virtual addressing, the memory will be both mapped and portable.
            //On a system with no unified virtual addressing, the memory will be neither mapped nor portable.
            cuda_error_check(cudaHostRegister(host_buffer,n*sizeof(T),cudaHostRegisterDefault));
        }else{
            host_buffer = make_host_locked_buffer<value_type>(n).release();
        }
        void* p;
        cuda_error_check(cudaHostGetDevicePointer(&p, host_buffer, 0));
        return pointer{static_cast<T*>(p)};
    }
    void deallocate(pointer p, size_type){
        if (host_data){
            cuda_error_check(cudaHostUnregister(host_data));
        }else{
            cuda_error_check(cudaFreeHost(ptr_to_void(p)));
        }
    }
    bool operator==(const cuda_mapping_allocator& other)const{return host_data == other.host_data;}
private:
    T* host_data;
};

}   //end of namespace cuda_experimental

#endif