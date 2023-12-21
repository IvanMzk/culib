#ifndef BENCHMARK_HELPERS_HPP_
#define BENCHMARK_HELPERS_HPP_

#include <array>
#include <sstream>
#include <iostream>
#include <numeric>
#include "cuda_helpers.hpp"

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
auto size_in_mbytes(std::size_t n){return size_in_bytes<T>(n)/double{1000000};}
template<typename T>
auto size_in_gbytes(std::size_t n){return size_in_bytes<T>(n)/double{1000000000};}
template<typename T>
auto size_to_str(std::size_t n){
    std::stringstream ss{};
    ss<<size_in_mbytes<T>(n)<<"MByte";
    return ss.str();
}
template<typename T>
auto bandwidth_to_str(std::size_t n, double dt_ms){
    std::stringstream ss{};
    ss<<size_in_mbytes<T>(n)/dt_ms<<"GBytes/s";
    return ss.str();
}

template<typename Timer>
class time_interval
{
    Timer start_{};
    Timer stop_{};
public:
    time_interval() = default;
    void start(){
        start_=Timer{};
    }
    void stop(){
        stop_=Timer{};
    }
    auto interval()const{
        return stop_-start_;
    }
    operator double()const{
        return interval();
    }
};

using cuda_interval = time_interval<culib::cuda_timer>;
using cpu_interval = time_interval<culib::cpu_timer>;

#if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__)
template<typename T> void fake_use(const T& t){asm volatile("":"+g"(const_cast<T&>(t)));}
#elif defined(_MSC_VER)
extern void msvc_fake_use(void*);
template<typename T> void fake_use(const T& t){msvc_fake_use(&const_cast<T&>(t));}
#endif

template<typename F, typename...Args, typename Interval=cuda_interval>
auto timing(F&& f, Args&&...args){
    Interval dt{};
    dt.start(),fake_use(f(std::forward<Args>(args)...)),dt.stop();
    return dt;
}

template<typename Container>
auto sum(const Container& intervals){
    using value_type = typename Container::value_type;
    return std::accumulate(intervals.begin(),intervals.end(),value_type{0});
}
template<typename Container>
auto mean(const Container& intervals){
    using value_type = typename Container::value_type;
    return sum(intervals)/intervals.size();
}
template<typename Container>
auto stdev(const Container& intervals){
    using value_type = typename Container::value_type;
    auto m = mean(intervals);
    auto v = std::accumulate(intervals.begin(),intervals.end(),value_type{0},[m](const auto& init, const auto& e){auto d=e-m; return init+d*d;})/intervals.size();
    return std::sqrt(v);
}

template<typename T, typename Size, typename Container>
auto statistic(const Size& size, const Container& intervals){
    std::stringstream ss{};
    const auto total_time = sum(intervals);
    const auto total_size = size*intervals.size();
    ss<<size_to_str<T>(size)<<" ";
    ss<<bandwidth_to_str<T>(total_size,total_time);
    return ss.str();
}


}   //end of namespace benchmark_helpers
#endif