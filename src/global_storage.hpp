#ifndef GLOBAL_STORAGE_HPP_
#define GLOBAL_STORAGE_HPP_

template<typename HostStorT, typename DevStorT>
class global_storage
{
    using host_storage_type = HostStorT;
    using device_storage_type = DevStorT;

    void to_host();
    void to_device(int device_id = 0);
    void free_host();
    void free_device();
    auto size()const;
    auto empty()const;

};

#endif