import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np

a = np.random.randn(4, 4).astype(np.float32)

print a

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

func(cuda.InOut(a), block=(4,4,1))

print a