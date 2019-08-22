# Custom OP 编译总结
测试编译了cpu及gpu环境下简单的custom op

## 系统环境
编译custom op的环境与编译tensorflow dll文件的环境相同

## 编译流程
具体编译流程请参阅各custom op路径下的readme文件

## 编译总结
* 编译gpu custom op时注意需在BUILD文件中添加gpu-srcs（对应于.cu.cc文件）
* 第一次编译custom op时需要在tensorflow根路径下重新运行configure.py以调试bazel编译器</br>
`python configure.py`
* 编译出的custom op dll文件在python中的调用请参阅[tensorflow官方指南](https://tensorflow.google.cn/guide/extend/op)

## Q&A
* Q： 编译出的dll文件怎样在c++中调用？
A： 暂时未找到成功在c++中调用custom op dll的方式，找到[可参考的解决方案](https://stackoverflow.com/questions/50125889/c-tensorflow-api-with-tensorrt/50449271#50449271)，但执行仍不成功

# add helpful link url here
[How to run custom GPU tensorflow::op from C++ code?](https://www.e-learn.cn/content/wangluowenzhang/995785)