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
* Cuda内核不支持4*4以上规模的矩阵逆运算（测试结果显示，但未找到官方证实），查找到部分资料，但测试结果仍不理想:</br>
	[Eigen cuda support](http://eigen.tuxfamily.org/dox/TopicCUDA.html)</br>
	[EIGEN_DEFAULT_DENSE_INDEX_TYPE explication](https://stackoverflow.com/questions/39685899/overload-resolution-of-eigens-operator-when-wrapping-it-with-boost-python/39691267#39691267)</br>
	[CUDA kernel can't support 8*8 eigen matrix assignment?](https://stackoverflow.com/questions/57504283/cuda-kernel-cant-support-88-eigen-matrix-assignment)</br>
