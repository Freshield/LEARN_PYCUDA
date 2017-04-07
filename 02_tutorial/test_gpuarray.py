import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

a = np.random.randn(4, 4).astype(np.float32)

a_gpu = gpuarray.to_gpu(a)

a_doubled = (a_gpu * 2)

print a_doubled

print a_gpu

