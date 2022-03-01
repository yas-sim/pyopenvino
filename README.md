# PyOpenVINO - An Experimental Python Implementation of OpenVINO inference engine (minimum-set)
----------------------------

## Description
The PyOpenVINO is a spin-off product from my deep learning algorithm study work. This project is not aiming neither practical performance nor rich functionarities.  
PyOpenVINO can load an OpenVINO IR model (.xml/.bin) and run it.  
The implementation is quite straightforward and naive. No Optimization technique is used. Thus, the code is easy to read and modify.  
Supported API is quite limited but it mimics OpenVINO IE Python API. So, you can easily read and modify the sample code too.  
- Developed as a spin-off from my deep learning study work.  
- **Very slow and limited functionality.** Not a general DL inference engine.  
- Simple and naive code: (I hope) This is a good reference for whom learning deep-learning technology.  
- Extensible ops: Ops are implemented as plugin. You can easily add your own ops as needed.  

------------------------

## How to run

Step 1 and 2 are optional since the converted MNIST IR model is provided.  

1. (Optional) Train a model and generate a '`saved_model`' with TensorFlow  
```sh
python mnist-tf-training.py
```
The trained model data will be created under `./mnist-savedmodel` directory.

2. (Optional) Convert TF saved_model into OpenVINO IR model  
Prerequisite: You need to have OpenVINO installed (Model Optimizer is required).  
```sh
convert-model.bat
```
Converted IR model (.xml/.bin) will be generated in `./models` directory.  

3. Run pyOpenVINO sample program
```sh
python test_pyopenvino.py
```
You'll see the output like this.  
```sh
pyopenvino>python test_pyopenvino.py
2.4045448303222656 sec/inf
[2 0 1 7 8 6 3 4 5 9] {'Func/StatefulPartitionedCall/output/_11:0': array([[7.8985232e-07, 2.0382242e-08, 9.9999917e-01, 1.0367380e-10,
        1.0184052e-10, 1.6024986e-12, 2.0729658e-10, 1.6014939e-08,
        6.5354605e-10, 9.5946288e-14]], dtype=float32)}
```

----------------------------------
## A Littile Description of the Implementation  

### IR model internal representation
Inference engine is using `networkx` as the internal representation of the IR model.
IR model will be translated into `node`s and `edge`s.  
The nodes represents the `ops` and the node holds the attributes of the ops (e.g. strides, dilates, etc).  
The edges represents the connection between the nodes. The edges holds the port number for both ends. Also, the edge holds the output data from the source nodes (ops).

### Ops plugins
Operators are implemented as plugin. You can develop an Op in Python and place the file in the `op_plugins` directory. The inference_engine of pyOpenVINO will search the Python source files in the `op_plugins` directory at the start time and register them as the Ops plugin.  
The file name of the Ops plugin will be treated as the Op name, so it must match the `layer type` attribute field in the IR XML file.  
The inference engine will call `compute()` function of the plugin to perform calculation.  The `compute()` function is the only API between the inference engine and the plugin. The inference engine will collect the required input data and pass it to the `compute()` function. The input data is in a form of Python `dict`.  
The op needs to calculate the result from the input data and return it as a Python `dict`.  

### Numpy version and Naive version  
Not all but some Ops have dual kernel implementation, a naive implementation and a numpy version (a bit faster) implementation.  

END