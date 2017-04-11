import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy

# create an array of 1s
data1 = ['Apple 12345 this is a long string again','Ding dong, it"s Pyton!!','digity! gigity!']
lines = numpy.array( data1, dtype=str)
linesGPU = cuda.mem_alloc(lines.size * lines.dtype.itemsize)
cuda.memcpy_htod(linesGPU, lines)
blocks = len(data1)
threadsPerBlock = lines.dtype.itemsize
nbr_values = lines.size * lines.dtype.itemsize # blocks * block_size
print("lines size: " + str(lines.size) + " itemsize : " + str(lines.dtype.itemsize))

# create a destination array that will receive the result
dest = numpy.zeros((nbr_values,), dtype=numpy.str)
destGPU =  cuda.mem_alloc(dest.size * dest.dtype.itemsize)

mod = SourceModule("""
__global__ void process(char **dest, char **line)
{
  int tID = threadIdx.x ;//+ blockIdx.x * blockDim.x;
  dest[tID] = line[tID];
}
""")

#Run the sourc model
gpusin = mod.get_function("process")
gpusin(destGPU, linesGPU, grid=(blocks,1), block=(threadsPerBlock,1,1))
cuda.memcpy_dtoh(dest, destGPU)
print str(len(dest))
print dest