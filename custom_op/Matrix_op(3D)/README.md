# Eigen Matrix Operation Test (3D input)
<br>

## Brief Discription
This program tests simple Eigen matrix operations in CUDA kernel, and proves that CUDA kernel supports Eigen structure and operations. (Based on tensorflow example program 'Add One') 
<br>

## Build instruction
* 1 Copy <code>BUILD</code> , <code>cuda_op_kernel.cu.cc</code>, and <code>cuda_op_kernel.cc</code> files to
> tensorflow\tensorflow\core\user_ops

* 2 Return to tensorflow root directory

* 3 Run 
> bazel build --config=opt --config=cuda //tensorflow/core/user_ops:AddOne.dll

* 4 Include this dll in python program and run it