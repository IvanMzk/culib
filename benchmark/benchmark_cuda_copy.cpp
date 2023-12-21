#include <numeric>
#include "catch.hpp"
#include "cuda_copy.hpp"
#include "benchmark_helpers.hpp"


namespace benchmark_cuda_copy{

using value_type = double;
using device_alloc_type = culib::device_allocator<value_type>;
using host_alloc_type = std::allocator<value_type>;
using culib::cuda_copy::copier;
using benchmark_helpers::timing;
using benchmark_helpers::statistic;

template<typename Copier>
struct benchmark_cuda_copy_helper{
    using copier_type = Copier;

    template<typename Size, typename DeviceIt, typename HostIt>
    auto operator()(const Size& size, const Size& n_iters, HostIt host_first, HostIt host_last, DeviceIt device_first, DeviceIt device_last){
        using value_type = typename std::iterator_traits<HostIt>::value_type;
        auto do_copy = [](auto first, auto last, auto dfirst){
            copier_type::copy(first,last,dfirst);
            return *dfirst;
        };
        std::vector<double> intervals_to_device{};
        std::vector<double> intervals_to_host{};
        for (std::size_t i{0}; i!=n_iters; ++i){
            std::iota(host_first, host_last, value_type{0});
            intervals_to_device.push_back(timing(do_copy,host_first,host_last,device_first));
            std::fill(host_first,host_last,0);
            intervals_to_host.push_back(timing(do_copy,device_first,device_last,host_first));
        }
        std::cout<<std::endl<<"to device "<<statistic<value_type>(size,intervals_to_device)<<" to host "<<statistic<value_type>(size,intervals_to_host);;
    }
};

template<typename CopierTag, typename Sizes, typename Size>
auto benchmark_cuda_copy_host_pointer(std::string mes, CopierTag, const Sizes& sizes, const Size& n_iters){
    using copier_type = copier<CopierTag>;
    device_alloc_type device_alloc{};
    host_alloc_type host_alloc{};
    std::cout<<std::endl<<"benchmark cuda_copy "<<mes;
    for (const auto& size : sizes){
        auto host_first = host_alloc.allocate(size);
        auto host_last = host_first+size;
        auto device_first = device_alloc.allocate(size);
        auto device_last = device_first+size;
        benchmark_cuda_copy_helper<copier_type>{}(size,n_iters,host_first,host_last,device_first,device_last);
        device_alloc.deallocate(device_first,size);
        host_alloc.deallocate(host_first,size);
    }
}

template<typename CopierTag, typename Sizes, typename Size>
auto benchmark_cuda_copy_host_iterator(std::string mes, CopierTag, const Sizes& sizes, const Size& n_iters){
    using copier_type = copier<CopierTag>;
    using container_type = std::vector<value_type>;
    device_alloc_type device_alloc{};
    std::cout<<std::endl<<"benchmark cuda_copy "<<mes;
    for (const auto& size : sizes){
        container_type host_container(size);
        auto host_first = host_container.begin();
        auto host_last = host_container.end();
        auto device_first = device_alloc.allocate(size);
        auto device_last = device_first+size;
        benchmark_cuda_copy_helper<copier_type>{}(size,n_iters,host_first,host_last,device_first,device_last);
        device_alloc.deallocate(device_first,size);
    }
}

}   //end of namespace benchmark_cuda_copy


TEST_CASE("benchmark_cuda_host_device_copier","[benchmark_cuda_copy]")
{
    using culib::cuda_copy::native_copier_tag;
    using culib::cuda_copy::multithread_copier_tag;
    using benchmark_helpers::make_sizes;
    using benchmark_cuda_copy::benchmark_cuda_copy_host_pointer;
    using benchmark_cuda_copy::benchmark_cuda_copy_host_iterator;

    constexpr std::size_t initial_size{1<<10};
    constexpr std::size_t factor{2};
    constexpr std::size_t n{19};
    constexpr auto sizes = make_sizes<initial_size,factor,n>();
    constexpr std::size_t n_iters{10};

    benchmark_cuda_copy_host_pointer("host pointer, native_copier_tag",native_copier_tag{},sizes,n_iters);
    benchmark_cuda_copy_host_pointer("host pointer, multithread_copier_tag",multithread_copier_tag{},sizes,n_iters);
    benchmark_cuda_copy_host_iterator("host iterator, native_copier_tag",native_copier_tag{},sizes,n_iters);
    benchmark_cuda_copy_host_iterator("host iterator, multithread_copier_tag",multithread_copier_tag{},sizes,n_iters);
}
