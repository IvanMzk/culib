#ifndef CUDA_HELPERS_HPP_
#define CUDA_HELPERS_HPP_

#include <stdexcept>
#include <sstream>
#include <iostream>
#include <chrono>
#include <thread>
#include "cuda_runtime.h"

namespace culib{

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
inline auto cuda_get_device_properties(int device){
    cudaDeviceProp prop;
    cuda_error_check(cudaGetDeviceProperties(&prop, device));
    return prop;
}
inline auto cuda_device_can_access_peer(int device, int peer_device){
    int i;
    cuda_error_check(cudaDeviceCanAccessPeer(&i,device,peer_device));
    return i;
}
inline auto cuda_stream_create(){
    cudaStream_t stream;
    cuda_error_check(cudaStreamCreate(&stream));
    return stream;
}
inline auto cuda_stream_destroy(cudaStream_t stream){
    cuda_error_check(cudaStreamDestroy(stream));
}
inline auto cuda_event_create(){
    cudaEvent_t event;
    cuda_error_check(cudaEventCreate(&event));
    return event;
}
inline auto cuda_event_destroy(cudaEvent_t event){
    cuda_error_check(cudaEventDestroy(event));
}

class device_switcher
{
    int device_to_switch_back;
public:
    device_switcher(int device_to_switch):
        device_to_switch_back{cuda_get_device()}
    {
        cuda_set_device(device_to_switch);
    }
    ~device_switcher(){cuda_set_device(device_to_switch_back);}
};

class cuda_stream
{
    cudaStream_t stream;
    bool sync_on_destruction;
    cuda_stream(const cuda_stream&) = delete;
    cuda_stream& operator=(const cuda_stream&) = delete;
public:
    ~cuda_stream(){
        if (stream){
            if (sync_on_destruction){
                cuda_error_check(cudaStreamSynchronize(stream));
            }
            cuda_stream_destroy(stream);
        }
    }
    cuda_stream(cuda_stream&& other):
        stream{other.stream},
        sync_on_destruction{other.sync_on_destruction}
    {
        other.stream = 0;
    }
    cuda_stream& operator=(cuda_stream&& other){
        stream = other.stream;
        sync_on_destruction = other.sync_on_destruction;
        other.stream = 0;
        return *this;
    }
    cuda_stream(bool sync_on_destruction_ = true):
        stream{cuda_stream_create()},
        sync_on_destruction{sync_on_destruction_}
    {}
    operator cudaStream_t(){return stream;}
    auto get()const{return stream;}
};

class cuda_timer
{
    cudaEvent_t event;
    cuda_timer(const cuda_timer&) = delete;
    cuda_timer& operator=(const cuda_timer&) = delete;
public:
    cuda_timer(cuda_timer&& other):
        event{other.event}
    {
        other.event = 0;
    }
    cuda_timer& operator=(cuda_timer&& other){
        event = other.event;
        other.event = 0;
        return *this;
    }
    ~cuda_timer(){
        if (event){
            cuda_event_destroy(event);
        }
    }
    cuda_timer(cudaStream_t stream = cudaStreamLegacy):
        event{cuda_event_create()}
    {
        cuda_error_check(cudaEventRecord(event, stream));
    }

    friend auto operator-(const cuda_timer& end, const cuda_timer& start){
        cuda_error_check(cudaEventSynchronize(start.event));
        cuda_error_check(cudaEventSynchronize(end.event));
        float dt_ms;
        cuda_error_check(cudaEventElapsedTime(&dt_ms,start.event,end.event));
        return dt_ms;
    }
};

class cpu_timer
{
    using clock_type = std::chrono::steady_clock;
    using time_point = typename clock_type::time_point;
    time_point point_;
public:
    cpu_timer():
        point_{clock_type::now()}
    {}
    friend auto operator-(const cpu_timer& end, const cpu_timer& start){
        return std::chrono::duration<float,std::milli>(end.point_-start.point_).count();
    }
};

class thread_sync_wrapper
{
    std::thread thread_;
public:
    ~thread_sync_wrapper(){
        if (thread_.joinable()){
            thread_.join();
        }
    }
    thread_sync_wrapper() = default;
    thread_sync_wrapper(thread_sync_wrapper&&) = default;
    thread_sync_wrapper& operator=(thread_sync_wrapper&&) = default;
    template<typename F, typename...Args>
    thread_sync_wrapper(F&& f, Args&&...args):
        thread_{std::forward<F>(f),std::forward<Args>(args)...}
    {}
};

}   //end of namespace culib

#endif