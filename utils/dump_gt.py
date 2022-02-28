import pickle

node = 'StatefulPartitionedCall/sequential/max_pooling2d/MaxPool'
node = 'StatefulPartitionedCall/sequential/conv2d_1/Conv2D'
node = 'StatefulPartitionedCall/sequential/conv2d/Conv2D'

def disp_result(data):
    N,C,H,W = data.shape
    for c in range(C):
        print('C=', c)
        for h in range(H):
            for w in range(W):
                print('{:6.3f},'.format(data[0,c,h,w]), end='')
            print()

with open('mnist_featmap7.pickle', 'rb') as f:
    fmap = pickle.load(f)

disp_result(fmap[node][2])
