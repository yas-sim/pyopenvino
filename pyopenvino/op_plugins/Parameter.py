# Parameter
import common_def
import numpy as np

def name():
    print('Parameter')

def compute(node:dict, inputs:dict=None, kernel_type:str='naive', debug:bool=False):
    if debug:
        print(node)
    shape = node['data']['shape']
    precision = common_def.type_convert_tbl[node['data']['element_type']]
    res = { 0 : np.array(node['param']).reshape(shape).astype(precision) }
    return res
