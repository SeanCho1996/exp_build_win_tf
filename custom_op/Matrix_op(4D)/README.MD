# Eigen Matrix Operation Test (4D input)
<br>

## Brief Discription
Based on what we've done in 3D condition, this time we simulated a more practical condition. For input, we import a 4D matrix with a component as batch, and modified a little on the calculate structure. As result, we achieved a correct 4D output.
<br>

## Build instruction
* 1 Copy <code>BUILD</code> , <code>eigen_4d.cu.cc</code>, and <code>eigen_4d.cc</code> files to
> tensorflow\tensorflow\core\user_ops

* 2 Return to tensorflow root directory

* 3 Run 
> bazel build --config=opt --config=cuda //tensorflow/core/user_ops:AddOne.dll

* 4 Include this dll in python program and run it