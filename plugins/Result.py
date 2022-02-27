# Result
import common_def

def name():
    print('Result')

def compute(node:dict, inputs:dict=None, debug:bool=False):
    if debug:
        print(node)

    # Validate input data
    for port, data in inputs.items():
        input_port = node['input'][port]
        if data.dtype != common_def.type_convert_tbl[input_port['precision']]:
            print('input data precision mismatch')
            return None
        if data.shape != input_port['dims']:
            print('input data shape mismatch')
            return None

    node['result'] = inputs[0]
    return []

# {'name': 'Func/StatefulPartitionedCall/output/_11:0', 'type': 'Result', 'version': 'opset1', 
# 'input': {0: {'precision': 'FP32', 'dims': (1, 10)}}}
