## Culib - library to work with `CUDA`using STL-like abstract types and algorithms.

Library provides:

- `device_pointer` class template abstraction, represents device memory address.
Its objects are dereferencible on host side, proxy object is returned.

- generic `copy` algorithm to copy data in any direction, supports copy in multiple threads.

- multithreading subsystem, including pool of reusable page-locked memory blocks and thread-pool to be used by `copy`.

- custom memcpy implementation that utilizes AVX can be used if standart implementation is too slow.

- `cuda_storage` class template abstraction represents device memory block and provide STL-like interface.

- device and page-locked memory allocators.

## License
This software is licensed under the [BSL 1.0](LICENSE.txt).