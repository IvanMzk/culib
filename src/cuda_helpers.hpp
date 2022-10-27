#ifndef CUDA_HELPERS_HPP_
#define CUDA_HELPERS_HPP_

#include <stdexcept>
#include <sstream>
#include "cuda_runtime.h"

namespace cuda_experimental{

class cuda_exception : public std::runtime_error{
    public: cuda_exception(const char* what):runtime_error(what){}
    public: cuda_exception(std::string what):runtime_error(what){}
};

#define cuda_error_check(ret) {cuda_assert(ret,__FILE__,__LINE__);}
inline void cuda_assert(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::stringstream ss{};
        ss<<"cuda runtime error, "<<cudaGetErrorString(err)<<" "<<file<<" "<<line;
        throw cuda_exception(ss.str());
    }
}
inline bool is_cuda_success(cudaError_t err){
    return err == cudaSuccess;
}
inline auto cuda_get_device(){
    int active_device;
    cuda_error_check(cudaGetDevice(&active_device));
    return active_device;
}
inline auto cuda_set_device(int dev){
    cuda_error_check(cudaSetDevice(dev));
}
inline auto cuda_get_device_count(){
    int n;
    cuda_error_check(cudaGetDeviceCount(&n));
    return n;
}



}   //end of namespace cuda_experimental

#endif