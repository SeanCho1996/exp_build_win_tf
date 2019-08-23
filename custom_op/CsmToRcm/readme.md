# Eigen 矩阵运算在cuda下的应用总结（对应CsmToRcm op的编译）

## 基本流程
在cuda中使用eigen矩阵操作的基本流程如下：</br>
__cpu部分__
* 提取input tensor
* 创建output tensor
* 将输入输出tensor转换为Eigen::TensorMap并做维度压缩（维度压缩在这一步并非必须，因为接下来还允许有维度变换操作）
* 将得到的tensormap映射成Eigen::Matrix(如果上一步没有做维度压缩，则必须在此处做压缩，因为Eigen::Matrix只能为二维)
* 调用cuda kernel，引用刚刚转换的输入输出matrix
</br>

__gpu部分__
* 分配线程，调用kernel函数
* kernel函数执行当前线程，在线程内对引用的输入matrix做矩阵运算
* 计算结果赋值到输出matrix

## Eigen矩阵运算的分析
在之前的测试工程中由于情况相对简单，运算量较小，所以运行无误，当前项目数据结构略复杂，运算量相对较大，因而发现了许多问题：</br>
* eigen矩阵内部默认为列组序，由tensormap向matrix转换时需要注意元素排列的方式（可以通过添加Eigen::Matrix::RowMajor修改为行组序）
* Cuda内核不支持4x4以上规模的矩阵逆运算（测试结果显示，但未找到官方证实），查找到部分资料，怀疑问题为cuda int型变量（32位）与host端int型变量（64位）不匹配，尝试基于此问题调整代码，但测试结果仍不理想:</br>
	[Eigen cuda support](http://eigen.tuxfamily.org/dox/TopicCUDA.html)</br>
	[sizeof Eigen::Index depends on system](http://www.alecjacobson.com/weblog/?p=4745)
	[EIGEN_DEFAULT_DENSE_INDEX_TYPE explication](https://stackoverflow.com/questions/39685899/overload-resolution-of-eigens-operator-when-wrapping-it-with-boost-python/39691267#39691267)</br>
	[CUDA kernel can't support 8*8 eigen matrix assignment?](https://stackoverflow.com/questions/57504283/cuda-kernel-cant-support-88-eigen-matrix-assignment)</br>
* Cuda内核似乎不支持复数矩阵求逆的操作（待核实）
* 发现eigen矩阵运算会占用cuda线程，kernel内部矩阵运算操作越多，可用的cuda线程就越少，解决方案为循环调用线程
* eigen延迟运算机制导致一些操作如果没有输出赋值，则该操作将不会被运算（如inverse，dot等）

## useful url link
[Eigen 官方网站](http://eigen.tuxfamily.org/index.php?title=Main_Page#Documentation)</br>
[Eigen unsupported module](http://eigen.tuxfamily.org/dox/unsupported/index.html)</br>
[Eigen operations (compare with Matlab)](https://blog.csdn.net/xuezhisdc/article/details/54645238)</br>
