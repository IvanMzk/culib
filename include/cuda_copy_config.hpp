#ifndef CUDA_COPPY_CONFIG_HPP_
#define CUDA_COPPY_CONFIG_HPP_

namespace culib{
namespace cuda_copy{

struct native_copier_tag{};
struct multithread_copier_tag{};

using copier_selector_type = multithread_copier_tag;    //select implementation of copy
inline constexpr bool native_memcpy = true; //culib implementation of memcpy_avx will be used if false
inline constexpr std::size_t memcpy_workers = 4;
inline constexpr std::size_t locked_pool_size = 4;
inline constexpr std::size_t locked_buffer_size = 64*1024*1024;
inline constexpr std::size_t locked_buffer_alignment = 4096;   //every buffer in locked pool must be aligned at least at locked_buffer_alignment, 0 - no alignment check
inline constexpr std::size_t multithread_threshold = 4*1024*1024;
inline constexpr std::size_t peer_multithread_threshold = 128*1024*1024;

}
}
#endif