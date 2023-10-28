import numpy as np
import timeit
import numba as nb
import skimage.io as skio
import math

def dual_tuple_division(x, y):
    return_tuple = []
    for i, ii in zip(x, y):
        return_tuple.append(math.ceil(ii/i))
    return tuple(return_tuple)

@nb.jit
def convert(src, dst):
    i = nb.threadIdx.x + nb.blockIdx.x * nb.blockDim.x
    j = nb.threadIdx.y + nb.blockIdx.y * nb.blockDim.y
    g = np.uint8((src[i, j, 0] + src[i, j, 1] + src[i, j, 2]) / 3)
    dst[i, j] = g
    
@nb.reduce
def find_max(a, b):
    if a > b:
        return a
    else:
        return b

@nb.reduce
def find_min(a, b):
    if a < b:
        return a
    else:
        return b

@nb.jit
def stretch(src, dst, min_g, max_g):
    i = nb.threadIdx.x + nb.blockIdx.x * nb.blockDim.x
    j = nb.threadIdx.y + nb.blockIdx.y * nb.blockDim.y
    dst[i, j] = np.uint8((src[i, j] - min_g) / (max_g - min_g) * 255)
        

block_size_list = [(2,2),
                   (4, 4),
                   (8, 8),
                   (16, 16), 
                   (32, 32)]

avgtime_list = []
for block_size in block_size_list:
    dtime_list = []
    for i in range(11):
        # Load and ignore alpha channel
        img = skio.imread('tiger.jpg')[:, :, :3]
        img = np.ascontiguousarray(img)
        h, w, _ = img.shape
        out = np.ones((h, w), dtype=img.dtype)

        # Configure Cuda blocks
        grid_size = dual_tuple_division(block_size, (h, w))

        # Measure time 
        stime = timeit.default_timer()
        # Map
        A = nb.to_device(img)
        B = nb.to_device(out)
        convert[grid_size, block_size](A, B)
        # Reduce
        TEMP = B.copy_to_host().flatten()
        min_g = find_min(TEMP)
        max_g = find_max(TEMP)
        # Map
        TEMP = TEMP.reshape((h, w))
        A = nb.to_device(TEMP)
        B = nb.to_device(np.array(TEMP, copy=True))
        stretch[grid_size, block_size](A, B, min_g, max_g)
        # Measure time
        dtime = timeit.default_timer() - stime
        dtime_list.append(dtime)
        
        out = B.copy_to_host()
        skio.imsave('out_gura.png', out)

    avgtime = sum(dtime_list[1:])/len(dtime_list[1:])
    avgtime_list.append(avgtime)
    print(f'{avgtime} @ {block_size}')