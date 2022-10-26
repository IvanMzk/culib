#include <iostream>
#include "catch.hpp"
#include "cuda_experimental.hpp"

namespace test_cuda_experimental{



}

TEST_CASE("test_pointer_attributes","[test_cuda_memory]"){
    using value_type = float;
    using cuda_mapping_allocator_type = cuda_experimental::cuda_mapping_allocator<value_type>;
    using cuda_allocator_type = cuda_experimental::cuda_allocator<value_type>;
    using cuda_experimental::unified_memory_allocator;
    using cuda_experimental::basic_pointer;
    using cuda_experimental::copy;
    using cuda_experimental::cuda_assert;
    using cuda_experimental::is_cuda_success;
    using cuda_experimental::make_host_buffer;

// enum __device_builtin__ cudaMemoryType
// {
//     cudaMemoryTypeUnregistered = 0, /**< Unregistered memory */
//     cudaMemoryTypeHost         = 1, /**< Host memory */
//     cudaMemoryTypeDevice       = 2, /**< Device memory */
//     cudaMemoryTypeManaged      = 3  /**< Managed memory */
// };

    auto print_ptr_attr = [](const auto& p){
        cudaPointerAttributes attr;
        auto err = cudaPointerGetAttributes(&attr, p);
        if (is_cuda_success(err)){
            std::cout<<std::endl<<"device"<<attr.device;
            std::cout<<std::endl<<"device_ptr"<<attr.devicePointer;
            std::cout<<std::endl<<"host_ptr"<<attr.hostPointer;
            switch (attr.type){
                case cudaMemoryType::cudaMemoryTypeUnregistered:
                    std::cout<<std::endl<<"Unregistered memory"<<attr.type;
                    break;
                case cudaMemoryType::cudaMemoryTypeHost:
                    std::cout<<std::endl<<"Host memory"<<attr.type;
                    break;
                case cudaMemoryType::cudaMemoryTypeDevice:
                    std::cout<<std::endl<<"Device memory"<<attr.type;
                    break;
                case cudaMemoryType::cudaMemoryTypeManaged:
                    std::cout<<std::endl<<"Managed memory"<<attr.type;
                    break;
            }
        }else{
            switch (err){
                case cudaErrorInvalidValue:
                    std::cout<<std::endl<<"cudaErrorInvalidValue"<<err;
                    break;
                case cudaErrorInvalidDevice:
                    std::cout<<std::endl<<"cudaErrorInvalidDevice"<<err;
                    break;
                default:
                    std::cout<<std::endl<<"cudaError???"<<err;
            }
        }
    };
    auto print_cuda_ptr_attr = [&](const auto& p){
        print_ptr_attr(p.get());
    };

    int n{100};
    int offset{99};
    //host locked
    auto mapping_alloc = cuda_mapping_allocator_type{};
    auto p = mapping_alloc.allocate(n);
    print_cuda_ptr_attr(p+offset);

    //host paged
    auto buffer_to_register = make_host_buffer<value_type>(n);
    auto mapping_alloc_registered = cuda_mapping_allocator_type{buffer_to_register.get()};
    auto p_registered = mapping_alloc_registered.allocate(n);
    print_cuda_ptr_attr(p_registered+offset);

    //dev
    auto dev_alloc = cuda_allocator_type{};
    auto p_dev = dev_alloc.allocate(n);
    print_cuda_ptr_attr(p_dev+offset);

    //host not registered in UVA
    auto buffer = make_host_buffer<value_type>(n);
    print_ptr_attr(buffer.get()+offset);

    //UM
    auto um_alloc = unified_memory_allocator<value_type>{};
    auto um_ptr = um_alloc.allocate(n);
    print_cuda_ptr_attr(um_ptr+offset);

    um_alloc.deallocate(um_ptr,n);
    mapping_alloc.deallocate(p,n);
    mapping_alloc_registered.deallocate(p_registered,n);
    dev_alloc.deallocate(p_dev,n);

}