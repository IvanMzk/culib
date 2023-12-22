## Culib - library to work with `CUDA`, using STL-like abstract types and algorithms.

Library provides:

- `device_pointer` class template abstraction, represents device memory address.
Such a pointer can be dereferenced on host side - proxy object is returned.

- generic `copy` algorithm to copy data in any direction, supports copy in multiple threads.

- multithreading subsystem, including pool of reusable page-locked memory blocks and thread-pool to be used by `copy`.

- custom `memcpy` implementation that utilizes AVX can be used if standart implementation is too slow.

- `cuda_storage` class template abstraction represents device memory block and provides STL-like interface.

- device and page-locked memory allocators.

## Including into project

Cmake `add_subdirectory(...)` command can be used to bring `culib` into your project:

```cmake
cmake_minimum_required(VERSION 3.5)
project(my_project)
add_subdirectory(path_to_culib_dir culib)
add_executable(my_target)
target_link_libraries(my_target PRIVATE culib)
...
```

## Usage

```cpp
#include <iostream>
#include "cuda_storage.hpp"

int main(int argc, const char* argv[]){

    const auto n = 1<<20;
    //select device to allocate memory block
    culib::cuda_set_device(0);

    //initialization
    //from init list
    culib::cuda_storage<double> stor0{1,2,3,4,5,6,7,8,9,10};
    std::cout<<std::endl<<*stor0.begin();   //1
    std::cout<<std::endl<<*(stor0.begin()+3);   //4
    *stor0.begin() = 1.1;
    std::cout<<std::endl<<*stor0.begin();   //1.1

    //from iterators range
    std::vector<double> vec(n,0);
    std::iota(vec.begin(),vec.end(),0.5);
    culib::cuda_storage<double> stor1(vec.begin(),vec.end());
    std::cout<<std::endl<<*(stor1.begin()+1234);    //1234.5

    //from size and value
    culib::cuda_storage<double> stor2(n);
    std::cout<<std::endl<<*(stor2.begin()+1234);    //any
    culib::cuda_storage<double> stor3(n,2.2);
    std::cout<<std::endl<<*(stor3.begin()+1234);    //2.2

    //copy
    //device to host
    std::vector<double> vec1(stor1.size(),0.0);
    copy(stor1.begin(),stor1.end(),vec1.begin());
    std::cout<<std::endl<<*(vec1.begin()+1234); //1234.5
    //host to device
    copy(vec1.begin(),vec1.end(),stor2.begin());
    std::cout<<std::endl<<*(stor2.begin()+1234);    //1234.5
    //device to device
    copy(stor2.begin(),stor2.end(),stor3.begin());
    std::cout<<std::endl<<*(stor3.begin()+1234);    //1234.5

    //fill
    fill(stor3.begin(),stor3.begin()+100,0.0);
    std::cout<<std::endl<<*stor3.begin();   //0.0

    //clone
    auto stor2_copy = stor2.clone();
    //clone to peer
    culib::cuda_set_device(1);
    auto stor2_copy_peer = stor2.clone();

    //clear - deallocates device memory block
    stor2_copy.clear();

    return 0;
}
```

## Build tests and benchmarks

[Catch](https://github.com/catchorg/Catch2) framework is used for testing.

To build and run tests:

```cmake
cmake -B build_dir -DBUILD_TEST=ON
cmake --build build_dir
build_dir/test/Test
```

To build and run benchmarks:

```cmake
cmake -B build_dir -DBUILD_BENCHMARK=ON
cmake --build build_dir
build_dir/benchmark/Benchmark
```

## License
This software is licensed under the [BSL 1.0](LICENSE.txt).