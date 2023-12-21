/*
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#ifndef CUDA_MEMORY_HPP_
#define CUDA_MEMORY_HPP_

#include "cuda_copy.hpp"

namespace culib{

//copy
//pageable to device
template<typename T>
auto copy(T* first, T* last, device_pointer<std::remove_const_t<T>> d_first){
    return cuda_copy::copier<cuda_copy::copier_selector_type>::copy(first,last,d_first);
}
template<typename It, typename T, std::enable_if_t<!culib::detail::is_basic_pointer_v<It>, int> =0>
auto copy(It first, It last, device_pointer<T> d_first){
    return cuda_copy::copier<cuda_copy::copier_selector_type>::copy(first,last,d_first);
}
//device to pageable
template<typename T>
auto copy(device_pointer<T> first, device_pointer<T> last, std::remove_const_t<T>* d_first){
    return cuda_copy::copier<cuda_copy::copier_selector_type>::copy(first,last,d_first);
}
template<typename T, typename It, std::enable_if_t<!culib::detail::is_basic_pointer_v<It>,int> =0>
auto copy(device_pointer<T> first, device_pointer<T> last, It d_first){
    return cuda_copy::copier<cuda_copy::copier_selector_type>::copy(first,last,d_first);
}
//device device
template<typename T>
auto copy(device_pointer<T> first, device_pointer<T> last, device_pointer<std::remove_const_t<T>> d_first){
    return cuda_copy::copier<cuda_copy::copier_selector_type>::copy(first,last,d_first);
}

//fill
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