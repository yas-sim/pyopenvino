import pickle
import sys
sys.path.append('pyopenvino')

# (node, inputs) are stored in a pickle file
with open('resources/node_args_6.pickle', 'rb') as f:
    node, inputs = pickle.load(file=f)

# ------------------------------------------------------------

import op_plugins.Convolution as op

print('Node name={} ,type={} is running.'.format(node['name'], node['type']))
res = op.compute(node, inputs)

print(res)
