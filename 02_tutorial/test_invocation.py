import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np

a = np.random.randn(4, 4).astype(np.float32)

a_gpu = cuda.mem_alloc(a.nbytes)

cuda.memcpy_htod(a_gpu, a)

mod = SourceModule(
    """
    __global__ void doublify(float *a)
    {
        int idx = threadIdx.x + threadIdx.y * 4;
        a[idx] *= 2;
    }
    """
)

func = mod.get_function("doublify")

grid = (1, 1)
block = (4, 4, 1)
func.prepare("P")
func.prepared_call(grid, block, a_gpu)

a_doubled = np.zeros_like(a)

cuda.memcpy_dtoh(a_doubled, a_gpu)

print a
print a_doubled