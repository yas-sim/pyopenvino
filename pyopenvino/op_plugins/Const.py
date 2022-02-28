# Const
import common_def
import numpy as np

def name():
    print('Const')

def compute(node:dict, inputs:dict=None, debug:bool=False):
    if debug:
        print(node)
    shape = node['data']['shape']
    precision = common_def.type_convert_tbl[node['data']['element_type']]
    res = { 0: np.array(node['const']['data'], dtype=precision).reshape(shape) }
    return res
