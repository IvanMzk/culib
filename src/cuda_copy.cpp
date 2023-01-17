#include "cuda_copy.hpp"

namespace cuda_experimental{
namespace cuda_memcpy{

inline constexpr std::size_t unrolling_factor = 4;

//256 block, aligned nt load and store
inline void copyn_avx_lasa(const __m256i*& first, std::size_t n, __m256i*& d_first){
    for (; n>=unrolling_factor; n-=unrolling_factor,first+=unrolling_factor,d_first+=unrolling_factor){
        _mm256_stream_si256(d_first,_mm256_stream_load_si256(first));
        _mm256_stream_si256(d_first+1,_mm256_stream_load_si256(first+1));
        _mm256_stream_si256(d_first+2,_mm256_stream_load_si256(first+2));
        _mm256_stream_si256(d_first+3,_mm256_stream_load_si256(first+3));
    }
    switch (n){
        case 0:
            break;
        case 1:
            _mm256_stream_si256(d_first++,_mm256_stream_load_si256(first++));
            break;
        case 2:
            _mm256_stream_si256(d_first++,_mm256_stream_load_si256(first++));
            _mm256_stream_si256(d_first++,_mm256_stream_load_si256(first++));
            break;
        case 3:
            _mm256_stream_si256(d_first++,_mm256_stream_load_si256(first++));
            _mm256_stream_si256(d_first++,_mm256_stream_load_si256(first++));
            _mm256_stream_si256(d_first++,_mm256_stream_load_si256(first++));
            break;
        default :
            break;
    }
    _mm_sfence();
}
//256 block, unaligned load and aligned nt store
inline void copyn_avx_lusa(const __m256i*& first, std::size_t n, __m256i*& d_first){
    for (; n>=unrolling_factor; n-=unrolling_factor,first+=unrolling_factor,d_first+=unrolling_factor){
        _mm256_stream_si256(d_first,_mm256_loadu_si256(first));
        _mm256_stream_si256(d_first+1,_mm256_loadu_si256(first+1));
        _mm256_stream_si256(d_first+2,_mm256_loadu_si256(first+2));
        _mm256_stream_si256(d_first+3,_mm256_loadu_si256(first+3));
    }
    switch (n){
        case 0:
            break;
        case 1:
            _mm256_stream_si256(d_first++,_mm256_loadu_si256(first++));
            break;
        case 2:
            _mm256_stream_si256(d_first++,_mm256_loadu_si256(first++));
            _mm256_stream_si256(d_first++,_mm256_loadu_si256(first++));
            break;
        case 3:
            _mm256_stream_si256(d_first++,_mm256_loadu_si256(first++));
            _mm256_stream_si256(d_first++,_mm256_loadu_si256(first++));
            _mm256_stream_si256(d_first++,_mm256_loadu_si256(first++));
            break;
        default :
            break;
    }
    _mm_sfence();
}

//256 block, aligned nt load and store
inline void copy_avx_lasa(const __m256i*& first, const __m256i* const last, __m256i*& d_first){
    for (; first!=last; ++first,++d_first){
        //auto block = _mm256_stream_load_si256(first);
        _mm256_stream_si256(d_first,_mm256_stream_load_si256(first));
    }
    _mm_sfence();
}
//256 block, unaligned load and aligned nt store
inline void copy_avx_lusa(const __m256i*& first, const __m256i* const last, __m256i*& d_first){
    for (; first!=last; ++first,++d_first){
        //auto block = _mm256_loadu_si256(first);
        _mm256_stream_si256(d_first,_mm256_loadu_si256(first));
    }
    _mm_sfence();
}

void* memcpy_avx(void* dst_host, const void* src_host, std::size_t n){
    //always do nt store
    using block_type = avx_block_type;
    static constexpr std::size_t block_alignment = alignof(block_type);

    auto dst_aligned = align<block_alignment>(dst_host);
    auto src_aligned = align<block_alignment>(src_host);
    if (dst_host == dst_aligned){
        //auto src_it = reinterpret_cast<const block_type*>(src_aligned);
        auto src_it = reinterpret_cast<const block_type*>(src_host);
        auto blocks_n = n/sizeof(block_type);
        auto dst_it = reinterpret_cast<block_type*>(dst_aligned);
        auto last_chunk_n = n%sizeof(block_type);
        if (src_host == src_aligned){
            copyn_avx_lasa(src_it, blocks_n, dst_it);
        }else{
            copyn_avx_lusa(src_it, blocks_n, dst_it);
        }
        std::memcpy(dst_it, src_it, last_chunk_n);  //copy last chunk
    }else{
        auto src_offset = reinterpret_cast<std::uintptr_t>(src_host)%block_alignment;
        auto dst_offset = reinterpret_cast<std::uintptr_t>(dst_host)%block_alignment;
        auto first_chunk_n = block_alignment - dst_offset;
        if (src_offset == dst_offset){
            auto src_it = reinterpret_cast<const block_type*>(src_aligned);
            auto n_ = n - first_chunk_n;
            auto blocks_n = n_/sizeof(block_type);
            auto last_chunk_n = n_%sizeof(block_type);
            auto dst_it = reinterpret_cast<block_type*>(dst_aligned);
            copyn_avx_lasa(src_it, blocks_n, dst_it);
            std::memcpy(dst_host, src_host, first_chunk_n);   //copy first chunk
            std::memcpy(dst_it, src_it, last_chunk_n);  //copy last chunk
        }else{
            auto src_it = reinterpret_cast<const block_type*>(reinterpret_cast<std::uintptr_t>(src_host) + first_chunk_n);
            auto n_ = n - first_chunk_n;
            auto blocks_n = n_/sizeof(block_type);
            auto last_chunk_n = n_%sizeof(block_type);
            auto dst_it = reinterpret_cast<block_type*>(dst_aligned);
            copyn_avx_lusa(src_it, blocks_n, dst_it);
            std::memcpy(dst_host, src_host, first_chunk_n);   //copy first chunk
            std::memcpy(dst_it, src_it, last_chunk_n);  //copy last chunk
        }
    }
    return dst_host;
}

// void memcpy_avx(void* dst_host, const void* src_host, std::size_t n){
//     //always do nt store
//     using block_type = avx_block_type;
//     static constexpr std::size_t block_alignment = alignof(block_type);

//     auto dst_aligned = align<block_alignment>(dst_host);
//     auto src_aligned = align<block_alignment>(src_host);
//     if (dst_host == dst_aligned){
//         //auto src_it = reinterpret_cast<const block_type*>(src_aligned);
//         auto src_it = reinterpret_cast<const block_type*>(src_host);
//         auto src_end = src_it + n/sizeof(block_type);
//         auto dst_it = reinterpret_cast<block_type*>(dst_aligned);
//         auto last_chunk_n = n%sizeof(block_type);
//         if (src_host == src_aligned){
//             copy_avx_lasa(src_it, src_end, dst_it);
//         }else{
//             copy_avx_lusa(src_it, src_end, dst_it);
//         }
//         std::memcpy(dst_it, src_it, last_chunk_n);  //copy last chunk
//     }else{
//         auto src_offset = reinterpret_cast<std::uintptr_t>(src_host)%block_alignment;
//         auto dst_offset = reinterpret_cast<std::uintptr_t>(dst_host)%block_alignment;
//         auto first_chunk_n = block_alignment - dst_offset;
//         if (src_offset == dst_offset){
//             auto src_it = reinterpret_cast<const block_type*>(src_aligned);
//             auto n_ = n - first_chunk_n;
//             auto src_end = src_it + n_/sizeof(block_type);
//             auto last_chunk_n = n_%sizeof(block_type);
//             auto dst_it = reinterpret_cast<block_type*>(dst_aligned);
//             copy_avx_lasa(src_it, src_end, dst_it);
//             std::memcpy(dst_host, src_host, first_chunk_n);   //copy first chunk
//             std::memcpy(dst_it, src_it, last_chunk_n);  //copy last chunk
//         }else{
//             auto src_it = reinterpret_cast<const block_type*>(reinterpret_cast<std::uintptr_t>(src_host) + first_chunk_n);
//             auto n_ = n - first_chunk_n;
//             auto src_end = src_it + n_/sizeof(block_type);
//             auto last_chunk_n = n_%sizeof(block_type);
//             auto dst_it = reinterpret_cast<block_type*>(dst_aligned);
//             copy_avx_lusa(src_it, src_end, dst_it);
//             std::memcpy(dst_host, src_host, first_chunk_n);   //copy first chunk
//             std::memcpy(dst_it, src_it, last_chunk_n);  //copy last chunk
//         }
//     }
// }

}
}