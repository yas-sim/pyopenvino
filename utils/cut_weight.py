import struct
import numpy as np

#     element_type : f32
#     shape : (64, 32, 3, 3)
#     offset : 1280
#     size : 73728

with open('mnist.bin', 'rb') as f:
    dt = f.read()

offset = 1280
size = 73728

buffer = dt[offset:offset+size]
format = '<{}f'.format(size//4)

decoded = struct.unpack(format, buffer)

a = np.array(decoded, dtype=np.float32).reshape(64,32,3,3)

print(a)
