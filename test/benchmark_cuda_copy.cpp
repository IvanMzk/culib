#include <numeric>
#include "catch.hpp"
#include "cuda_copy.hpp"
#include "benchmark_helpers.hpp"


namespace benchmark_cuda_copy{

using cuda_experimental::device_pointer;
using cuda_experimental::cuda_stream;
using cuda_experimental::cuda_assert;

//host to device
template<typename T>
void copy_baseline(T* first, T* last, device_pointer<std::remove_const_t<T>> d_first){
    auto n = std::distance(first,last)*sizeof(T);
    cuda_error_check(cudaMemcpyAsync(d_first, first, n, cudaMemcpyKind::cudaMemcpyHostToDevice, cuda_stream{}));
}
//device to host
template<typename T>
void copy_baseline(device_pointer<T> first, device_pointer<T> last, std::remove_const_t<T>* d_first){
    auto n = distance(first,last)*sizeof(T);
    cuda_error_check(cudaMemcpyAsync(d_first, first, n, cudaMemcpyKind::cudaMemcpyDeviceToHost, cuda_stream{}));
}

using cuda_experimental::cuda_copy::copier;
using cuda_experimental::cuda_copy::multithread_copier_tag;
using cuda_experimental::cuda_copy::native_copier_tag;



}   //end of namespace benchmark_cuda_copy


TEMPLATE_TEST_CASE("benchmark_cuda_copier","[benchmark_cuda_copy]",
    (std::tuple<cuda_experimental::cuda_copy::copier<cuda_experimental::cuda_copy::native_copier_tag>,std::size_t>),
    (std::tuple<cuda_experimental::cuda_copy::copier<cuda_experimental::cuda_copy::multithread_copier_tag>,std::size_t>)
)
{
    using copier_type = std::tuple_element_t<0,TestType>;
    using value_type = std::tuple_element_t<1,TestType>;
    using benchmark_helpers::make_sizes;
    using benchmark_helpers::size_to_str;
    using benchmark_helpers::bandwidth_to_str;
    using cuda_experimental::cuda_timer;
    using device_alloc_type = cuda_experimental::device_allocator<value_type>;
    using host_alloc_type = std::allocator<value_type>;

    constexpr std::size_t initial_size{1<<10};
    constexpr std::size_t factor{2};
    constexpr std::size_t n{20};
    constexpr auto sizes = make_sizes<initial_size,factor,n>();
    constexpr std::size_t iters_per_size{10};

    device_alloc_type device_alloc{};
    host_alloc_type host_alloc{};
    std::cout<<std::endl<<"benchmark_cuda_copier "<<typeid(copier_type).name();
    for (const auto& size : sizes){
        float dt_ms_to_device{0};
        float dt_ms_to_host{0};
        for (std::size_t i{0}; i!=iters_per_size; ++i){
            auto device_ptr = device_alloc.allocate(size);
            auto host_src_ptr = host_alloc.allocate(size);
            std::iota(host_src_ptr, host_src_ptr+size, value_type{0});
            cuda_timer start_to_device{};
            copier_type::copy(host_src_ptr,host_src_ptr+size,device_ptr);
            cuda_timer stop_to_device{};
            dt_ms_to_device += stop_to_device - start_to_device;
            auto host_dst_ptr = host_alloc.allocate(size);
            std::fill(host_dst_ptr,host_dst_ptr+size,0);
            cuda_timer start_to_host{};
            copier_type::copy(device_ptr, device_ptr+size, host_dst_ptr);
            cuda_timer stop_to_host{};
            dt_ms_to_host += stop_to_host - start_to_host;
            REQUIRE(std::equal(host_src_ptr, host_src_ptr+size, host_dst_ptr));
            device_alloc.deallocate(device_ptr,size);
            host_alloc.deallocate(host_src_ptr,size);
            host_alloc.deallocate(host_dst_ptr,size);
        }
        std::cout<<std::endl<<size_to_str<value_type>(size)<<" to_device "<<bandwidth_to_str<value_type>(size*iters_per_size, dt_ms_to_device)<<
            " to_host "<<bandwidth_to_str<value_type>(size*iters_per_size, dt_ms_to_host)<<" total ms "<<dt_ms_to_device+dt_ms_to_host;
    }
}

TEMPLATE_TEST_CASE("benchmark_cuda_copier_iterator","[benchmark_cuda_copy]",
    (std::tuple<cuda_experimental::cuda_copy::copier<cuda_experimental::cuda_copy::native_copier_tag>,std::size_t>),
    (std::tuple<cuda_experimental::cuda_copy::copier<cuda_experimental::cuda_copy::multithread_copier_tag>,std::size_t>)
)
{

    using copier_type = std::tuple_element_t<0,TestType>;
    using value_type = std::tuple_element_t<1,TestType>;
    using benchmark_helpers::make_sizes;
    using benchmark_helpers::size_to_str;
    using benchmark_helpers::bandwidth_to_str;
    using cuda_experimental::cuda_timer;
    using device_alloc_type = cuda_experimental::device_allocator<value_type>;
    using host_alloc_type = std::allocator<value_type>;

    constexpr std::size_t initial_size{1<<10};
    constexpr std::size_t factor{2};
    constexpr std::size_t n{20};
    constexpr auto sizes = make_sizes<initial_size,factor,n>();
    constexpr std::size_t iters_per_size{10};
    using container_type = std::vector<value_type>;

    device_alloc_type device_alloc{};
    host_alloc_type host_alloc{};
    std::cout<<std::endl<<"benchmark_cuda_copier_iterator "<<typeid(copier_type).name();
    for (const auto& size : sizes){
        float dt_ms_to_device{0};
        float dt_ms_to_host{0};
        for (std::size_t i{0}; i!=iters_per_size; ++i){
            auto device_ptr = device_alloc.allocate(size);
            container_type host_src(size);
            std::iota(host_src.begin(), host_src.end(), value_type{0});
            cuda_timer start_to_device{};
            copier_type::copy(host_src.begin(),host_src.end(),device_ptr);
            cuda_timer stop_to_device{};
            dt_ms_to_device += stop_to_device - start_to_device;
            container_type host_dst(size);
            std::fill(host_dst.begin(), host_dst.end(),0);
            cuda_timer start_to_host{};
            copier_type::copy(device_ptr, device_ptr+size, host_dst.begin());
            cuda_timer stop_to_host{};
            dt_ms_to_host += stop_to_host - start_to_host;
            REQUIRE(std::equal(host_src.begin(), host_src.end(), host_dst.begin()));
            device_alloc.deallocate(device_ptr,size);
        }
        std::cout<<std::endl<<size_to_str<value_type>(size)<<" to_device "<<bandwidth_to_str<value_type>(size*iters_per_size, dt_ms_to_device)<<
            " to_host "<<bandwidth_to_str<value_type>(size*iters_per_size, dt_ms_to_host)<<" total ms "<<dt_ms_to_device+dt_ms_to_host;
    }
}