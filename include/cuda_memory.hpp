#ifndef CUDA_MEMORY_HPP_
#define CUDA_MEMORY_HPP_

#include "cuda_pointer.hpp"
#include "cuda_allocator.hpp"
#include "cuda_copy.hpp"

namespace culib{

template<typename T>
void fill(device_pointer<T> first, device_pointer<T> last, const T& v){
    auto n = static_cast<std::size_t>(distance(first,last));
    auto n_buf = cuda_copy::locked_buffer_size/sizeof(T);
    auto n_buf_bytes = n_buf*sizeof(T);
    auto buf = cuda_copy::locked_pool().pop();
    std::uninitialized_fill_n(reinterpret_cast<T*>(buf.get().data().get()),n_buf,v);
    for(; n>=n_buf; n-=n_buf, first+=n_buf){
        cuda_copy::dma_to_device(first,buf,n_buf_bytes);
    }
    if(n){
        cuda_copy::dma_to_device(first,buf,n*sizeof(T));
    }
}

}   //end of namespace culib{

#endif