
import numpy as np

format_config = { 'FP32': ['f', 4], 'FP16': ['e', 2], 'F32' : ['f', 4], 'F16' : ['e', 2],
                  'I64' : ['q', 8], 'I32' : ['i', 4], 'I16' : ['h', 2], 'I8'  : ['b', 1], 'U8'  : ['B', 1] }

type_convert_tbl = { 'f32':np.float32, 'f16':np.float16, 'i64':np.int64, 'i32':np.int32, 'i16':np.int16, 'i8':np.int8, 'u8':np.uint8,
                    'FP32':np.float32, 'FP16':np.float16 }

#type_convert_tbl = { 'f32':np.dtype[np.float32], 'f16':np.dtype[np.float16], 'i64':np.dtype[int64], 'i32':np.dtype[int32], 'i16':np.dtype[int16], 'i8':np.dtype[int8], 'u8':dtype[np.uint8],
#                    'FP32':np.dtype[np.float32], 'FP16':np.dtype[np.float16] }

def print_dict(dic:dict, indent_level=0, indent_step=4):
    for key, val in dic.items():
        print(' ' * indent_step * indent_level, key, ': ', end='')
        if type(val) is dict:
            print()
            print_dict(val, indent_level+1)
        else:
            print(val)

def string_to_tuple(string:str):
    tmp_list = [ int(item) for item in string.split(',') ]
    return tuple(tmp_list)