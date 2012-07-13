Copyright Christian Fobel 2012.
This project is released under the [LGPLv2.1] [lgpl]

This project implements a CUDA device function for eliminating
duplicate values from an array in shared memory.

Note that the order of the resulting elements will vary depending on
the scheduling of threads.  If an ordered list of unique elements is
required, a sort can be done on the resulting list.


Project contents:
-----------------

`unique/pycuda_include/`:

 * `unique/pycuda_include/unique.hpp`:
  * Header containing device function `int unique(int size, T *data)`
  * Note that the `unique` function is contained in the `unique`
    namespace
 * `unique/pycuda_include/bitonic_sort.hpp`
  * header containing sort used by `unique`

`unique/pycuda_templates/`:

 * `unique/pycuda_templates/unique.cu`
  * example CUDA kernel using `unique`

`unique/`:

 * `unique/cuda.py`
  * PyCUDA interface to example kernel

`tests/`:
 * nose test cases


Usage:
------

To use the `unique` device function, simply copy `unique.hpp` and
`bitonic_sort.hpp` to your project.  No other files are necessary.

[lgpl]: http://www.gnu.org/licenses/lgpl-2.1.html
