#ifndef BENCHMARK_HELPERS_HPP_
#define BENCHMARK_HELPERS_HPP_

#include <array>
#include <sstream>
#include <iostream>
#include "catch.hpp"
#include "cuda_memory.hpp"

namespace benchmark_helpers{

    template<std::size_t Init, std::size_t Fact>
    constexpr auto make_size_helper(std::size_t i){
        if (i==0){
            return Init;
        }else{
            return Fact*make_size_helper<Init,Fact>(i-1);
        }
    }
    template<std::size_t Init, std::size_t Fact, std::size_t...I>
    auto constexpr make_sizes_helper(std::index_sequence<I...>){
        return std::array<std::size_t, sizeof...(I)>{make_size_helper<Init,Fact>(I)...};
    }
    template<std::size_t Init = (1<<20), std::size_t Fact = 2, std::size_t N>
    auto constexpr make_sizes(){
        return make_sizes_helper<Init,Fact>(std::make_index_sequence<N>{});
    }

    template<typename T>
    auto size_in_bytes(std::size_t n){return n*sizeof(T);}
    template<typename T>
    auto size_in_mbytes(std::size_t n){return size_in_bytes<T>(n)/std::size_t{1000000};}
    template<typename T>
    auto size_in_gbytes(std::size_t n){return size_in_bytes<T>(n)/std::size_t{1000000000};}

    template<typename T>
    auto size_to_str(std::size_t n){
        std::stringstream ss{};
        ss<<size_in_mbytes<T>(n)<<"MByte";
        return ss.str();
    }
    template<typename T>
    auto bandwidth_to_str(std::size_t n, float dt_ms){
        std::stringstream ss{};
        ss<<size_in_mbytes<T>(n)/dt_ms<<"GBytes/s";
        return ss.str();
    }



    template<typename T>
    struct pageable_uninitialized_buffer_maker
    {
        using value_type = T;
        static constexpr char name[] = "pageable_uninitialized_buffer_maker";
        template<typename U>
        auto operator()(const U& n){return cuda_experimental::pageable_buffer<value_type>(n);}
    };
    template<typename T>
    struct pageable_initialized_buffer_maker
    {
        using value_type = T;
        static constexpr char name[] = "pageable_initialized_buffer_maker";
        value_type init_data_;
        pageable_initialized_buffer_maker() = default;
        pageable_initialized_buffer_maker(const value_type& init_data__):
            init_data_{init_data__}
        {}
        template<typename U>
        auto operator()(const U& n){
            auto buf = cuda_experimental::pageable_buffer<value_type>(n);
            std::uninitialized_fill(buf.begin(), buf.end(), init_data_);
            return buf;
        }
    };
    template<typename T>
    struct locked_buffer_maker
    {
        using value_type = T;
        static constexpr char name[] = "locked_buffer_maker";
        template<typename U>
        auto operator()(const U& n){return cuda_experimental::locked_buffer<value_type>(n);}
    };
    // template<typename T>
    // struct locked_write_combined_buffer_maker
    // {
    //     using value_type = T;
    //     static constexpr char name[] = "locked_write_combined_buffer_maker";
    //     template<typename U>
    //     auto operator()(const U& n){return cuda_experimental::make_locked_memory_buffer<value_type>(n,cudaHostAllocWriteCombined);}
    // };

}


#endif