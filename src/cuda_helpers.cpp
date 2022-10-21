#include "cuda_helpers.hpp"

namespace cuda_experimental{

void cuda_assert(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::stringstream ss{};
        ss<<"cuda runtime error, "<<cudaGetErrorString(err)<<" "<<file<<" "<<line;
        throw cuda_exception(ss.str());
    }
}

bool is_cuda_success(cudaError_t err){
    return err == cudaSuccess;
}

}   //end of namespace cuda_experimental
