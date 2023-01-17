#include "cuda_copy.hpp"

namespace cuda_experimental{
namespace cuda_memcpy{

void memcpy_avx(void* dst_host, const void* src_host, std::size_t n){
    //always do nt store
    using block_type = avx_block_type;
    static constexpr std::size_t block_alignment = alignof(block_type);

    auto dst_aligned = align<block_alignment>(dst_host);
    auto src_aligned = align<block_alignment>(src_host);
    if (dst_host == dst_aligned){
        //auto src_it = reinterpret_cast<const block_type*>(src_aligned);
        auto src_it = reinterpret_cast<const block_type*>(src_host);
        auto src_end = src_it + n/sizeof(block_type);
        auto dst_it = reinterpret_cast<block_type*>(dst_aligned);
        auto last_chunk_n = n%sizeof(block_type);
        if (src_host == src_aligned){
            std::cout<<std::endl<<"dst_aligned_src_aligned";
            for (; src_it!=src_end; ++src_it,++dst_it){
                //load aligned nt, store aligned nt
                auto block = _mm256_stream_load_si256(src_it);
                _mm256_stream_si256(dst_it,block);
            }
        }else{
            std::cout<<std::endl<<"dst_aligned_src_unaligned";
            for (; src_it!=src_end; ++src_it,++dst_it){
               //load unaligned, store to aligned nt
                auto block = _mm256_loadu_si256(src_it);
                _mm256_stream_si256(dst_it,block);
            }
        }
        std::memcpy(dst_it, src_it, last_chunk_n);  //copy last
    }else{
        auto src_offset = reinterpret_cast<std::uintptr_t>(src_host)%block_alignment;
        auto dst_offset = reinterpret_cast<std::uintptr_t>(dst_host)%block_alignment;
        auto first_chunk_n = block_alignment - dst_offset;
        if (src_offset == dst_offset){
            std::cout<<std::endl<<"src_dst_unaligned_equal_offset";
            auto src_it = reinterpret_cast<const block_type*>(src_aligned);
            auto n_ = n - first_chunk_n;
            auto src_end = src_it + n_/sizeof(block_type);
            auto last_chunk_n = n_%sizeof(block_type);
            auto dst_it = reinterpret_cast<block_type*>(dst_aligned);
            for (; src_it!=src_end; ++src_it,++dst_it){
                //load aligned nt, store aligned nt
                auto block = _mm256_stream_load_si256(src_it);
                _mm256_stream_si256(dst_it,block);
            }
            std::memcpy(dst_host, src_host, first_chunk_n);   //copy first
            std::memcpy(dst_it, src_it, last_chunk_n);  //copy last
        }else{
            auto src_it = reinterpret_cast<const block_type*>(reinterpret_cast<unsigned char>(src_host) + first_chunk_n);
            auto n_ = n - first_chunk_n;
            auto src_end = src_it + n_/sizeof(block_type);
            auto last_chunk_n = n_%sizeof(block_type);
            auto dst_it = reinterpret_cast<block_type*>(dst_aligned);
            for (; src_it!=src_end; ++src_it,++dst_it){
               //load unaligned, store to aligned nt
                auto block = _mm256_loadu_si256(src_it);
                _mm256_stream_si256(dst_it,block);
            }
            std::memcpy(dst_host, src_host, first_chunk_n);   //copy first
            std::memcpy(dst_it, src_it, last_chunk_n);  //copy last
        }
    }
}

}
}