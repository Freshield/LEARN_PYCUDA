import pycuda.reduction as rd
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import numpy as np

a = gpuarray.arange(400, dtype=np.float32)
b = gpuarray.arange(400, dtype=np.float32)

krnl = rd.ReductionKernel(np.float32, neutral='0',
                          reduce_expr='a+b', map_expr='x[i]*y[i]',
                          arguments='float *x, float *y')

my_dot_prod = krnl(a, b).get()

print my_dot_prod

print np.sum(np.arange(400) ** 2)