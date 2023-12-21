#include <numeric>
#include "catch.hpp"
#include "cuda_copy.hpp"
#include "cuda_algorithm.hpp"
#include "benchmark_helpers.hpp"


namespace benchmark_cuda_copy{

using value_type = double;
using device_alloc_type = culib::device_allocator<value_type>;
using host_alloc_type = std::allocator<value_type>;
using benchmark_helpers::timing;
using benchmark_helpers::statistic;

struct benchmark_cuda_copy_helper{
    template<typename Size, typename DeviceIt, typename HostIt, typename Command>
    auto operator()(const Size& size, const Size& n_iters, HostIt host_first, HostIt host_last, DeviceIt device_first, DeviceIt device_last, Command command){
        using value_type = typename std::iterator_traits<HostIt>::value_type;
        std::vector<double> intervals_to_device{};
        std::vector<double> intervals_to_host{};
        for (std::size_t i{0}; i!=n_iters; ++i){
            std::iota(host_first, host_last, value_type{0});
            intervals_to_device.push_back(timing(command,host_first,host_last,device_first));
            std::fill(host_first,host_last,0);
            intervals_to_host.push_back(timing(command,device_first,device_last,host_first));
        }
        std::cout<<std::endl<<"to device "<<statistic<value_type>(size,intervals_to_device)<<" to host "<<statistic<value_type>(size,intervals_to_host);;
    }
};

template<typename Sizes, typename Size, typename Command>
auto benchmark_cuda_copy_host_pointer(std::string mes, const Sizes& sizes, const Size& n_iters, Command command){
    device_alloc_type device_alloc{};
    host_alloc_type host_alloc{};
    std::cout<<std::endl<<mes;
    for (const auto& size : sizes){
        auto host_first = host_alloc.allocate(size);
        auto host_last = host_first+size;
        auto device_first = device_alloc.allocate(size);
        auto device_last = device_first+size;
        benchmark_cuda_copy_helper{}(size,n_iters,host_first,host_last,device_first,device_last,command);
        device_alloc.deallocate(device_first,size);
        host_alloc.deallocate(host_first,size);
    }
}

template<typename Sizes, typename Size, typename Command>
auto benchmark_cuda_copy_host_iterator(std::string mes, const Sizes& sizes, const Size& n_iters, Command command){
    using container_type = std::vector<value_type>;
    device_alloc_type device_alloc{};
    std::cout<<std::endl<<mes;
    for (const auto& size : sizes){
        container_type host_container(size);
        auto host_first = host_container.begin();
        auto host_last = host_container.end();
        auto device_first = device_alloc.allocate(size);
        auto device_last = device_first+size;
        benchmark_cuda_copy_helper{}(size,n_iters,host_first,host_last,device_first,device_last,command);
        device_alloc.deallocate(device_first,size);
    }
}

}   //end of namespace benchmark_cuda_copy

TEST_CASE("benchmark_cuda_host_device_copier","[benchmark_cuda_copy]")
{
    using culib::cuda_copy::native_copier_tag;
    using culib::cuda_copy::multithread_copier_tag;
    using culib::cuda_copy::copier;
    using benchmark_helpers::make_sizes;
    using benchmark_cuda_copy::benchmark_cuda_copy_host_pointer;
    using benchmark_cuda_copy::benchmark_cuda_copy_host_iterator;

    constexpr std::size_t initial_size{1<<10};
    constexpr std::size_t factor{2};
    constexpr std::size_t n{19};
    constexpr auto sizes = make_sizes<initial_size,factor,n>();
    constexpr std::size_t n_iters{1};

    auto copy_native = [](auto first, auto last, auto dfirst){
        using value_type = typename std::iterator_traits<decltype(dfirst)>::value_type;
        copier<native_copier_tag>::copy(first,last,dfirst);
        return static_cast<value_type>(*dfirst);
    };
    auto copy_multithread = [](auto first, auto last, auto dfirst){
        using value_type = typename std::iterator_traits<decltype(dfirst)>::value_type;
        copier<multithread_copier_tag>::copy(first,last,dfirst);
        return static_cast<value_type>(*dfirst);
    };

    benchmark_cuda_copy_host_pointer("bench culib::cuda_copy::copier, host pointer, native_copy",sizes,n_iters,copy_native);
    benchmark_cuda_copy_host_pointer("bench culib::cuda_copy::copier, host pointer, multithread_copy",sizes,n_iters,copy_multithread);
    benchmark_cuda_copy_host_iterator("bench culib::cuda_copy::copier, host iterator, native_copy",sizes,n_iters,copy_native);
    benchmark_cuda_copy_host_iterator("bench culib::cuda_copy::copier, host iterator, multithread_copy",sizes,n_iters,copy_multithread);
}

TEST_CASE("benchmark_cuda_host_device_copy_algorithm","[benchmark_cuda_copy]")
{
    using benchmark_helpers::make_sizes;
    using benchmark_cuda_copy::benchmark_cuda_copy_host_pointer;
    using benchmark_cuda_copy::benchmark_cuda_copy_host_iterator;

    constexpr std::size_t initial_size{1<<10};
    constexpr std::size_t factor{2};
    constexpr std::size_t n{19};
    constexpr auto sizes = make_sizes<initial_size,factor,n>();
    constexpr std::size_t n_iters{1};

    auto copy_algo = [](auto first, auto last, auto dfirst){
        using value_type = typename std::iterator_traits<decltype(dfirst)>::value_type;
        copy(first,last,dfirst);
        return static_cast<value_type>(*dfirst);
    };

    benchmark_cuda_copy_host_pointer("bench culib::copy, host pointer",sizes,n_iters,copy_algo);
    benchmark_cuda_copy_host_iterator("bench culib::copy, host iterator",sizes,n_iters,copy_algo);
}
