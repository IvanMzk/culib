## Culib - library to work with `CUDA`, using STL-like abstract types and algorithms.

Library provides:

- `device_pointer` class template abstraction, represents device memory address.
Its objects can be dereferenced on host side - proxy object is returned.

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

## Examples of usage



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