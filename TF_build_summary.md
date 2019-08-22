# Tensorflow .whl及.dll在windows环境下编译总结
## Tensorflow-cpu
测试了tensorflow-cpu whl文件的编译

### 系统环境：
* Windows 10
* Anaconda: 4.7.10
* Python: 3.6.7
* ~~CUDA: 10.0~~
* ~~CUDNN: 7.6.1~~
* Tensorflow: 1.9.1 cpu
* bazel: 0.16.0
* msvc: 2015 update 3

### 编译流程：
cpu版本的编译较为简单，请直接参阅[tensorflow官方文档](https://tensorflow.google.cn/install/source_windows "tensorflow官方文档")

### 编译总结：
cpu版本的编译为下一步编译gpu版本铺垫了基础，在编译过程中需要注意如下问题
* 执行编译的主机用户名中不能包含空格（bazel的编译实际上在`C:/users/用户名`中进行，bazel无法识别含空格的编译路径）
* MSVC2015建议安装完整的Visual Studio， 而不是只安装build tool
* bazel版本一定要注意匹配，过高或过低都无法完成编译

## Tensorflow-gpu
测试了tensorflow-gpu dll，whl文件的编译

### 系统环境：
请参阅[TF_build_from_source_on_windows.md](https://github.com/7oud/exp_build_win_tf/blob/master/TF_build_from_source_on_windows.md "TF_build_from_source_on_windows.md")

### 编译流程：
请参阅[TF_build_from_source_on_windows.md](https://github.com/7oud/exp_build_win_tf/blob/master/TF_build_from_source_on_windows.md "TF_build_from_source_on_windows.md")

### 编译总结：
此编译脚本引用自[guikarist](https://github.com/guikarist "guikarist")，更完整的编译流程请参阅[ta的原创脚本](https://github.com/guikarist/tensorflow-windows-build-script)</br>
在编译cpu的基础上，编译gpu版本时仍需注意如下问题：
* 测试编译whl文件时未能成功编译，理论上只需将编译选项中`-BazelBuildParameters`的最后一项修改为`//tensorflow/tools/pip_package:build_pip_package`，但编译未通过，提示问题为DLL文件加载失败，可能的原因为bazel与cuda，cudnn版本不匹配（测试版本为bazel 0.21.0， cuda v10.0， cudnn 7.6.2）
* 在编译过程中需要注意网络连接通畅，最好可以接入外网
* 利用当前脚本编译生成的include文件并不完全，只能满足最简单的测试功能，如需要更完善的include目录，请下载include.tar(待上传)
* 在visual studio中测试编译出的dll时，除添加库文件，引用文件位置及添加链接器外，还需注意修改“属性 - C/C++ - 预处理器 - 预处理器定义”处，添加“NOMINMAX”选项
* 在编译中途失败，或需要重新编译时，注意添加`-ReserveSource`编译选项以保存之前编译的部分结果

## Q&A
* Q: 脚本调用git命令时出现`BUG: run-command.c:500: disabling cancellation: Invalid argument`？</br>
A： 运行MSYS2.exe，无需在MSYS命令行输入任何指令，保持该窗口在后台运行，即可正常使用git命令
* Q：在编译过程中，出现类似`error loading package '': Encountered error while reading extension file 'repositories/repositories.bzl': no such package '@io_bazel_rules_docker//repositories':`的error？</br>
A： 在load package阶段出现类似错误通常原因为网络连接不稳定，建议更换网络或连接VPN后重新尝试编译
* Q：编译出的tensorflow c++ API不能调用第三方库（如libprotobuf)？</br>
A： 当前版本tensorflow环境（1.13.1）下，第三方库文件只能与其源代码一同编译使用，直接链接.lib或.a文件无法使用（[reference](https://github.com/guikarist/tensorflow-windows-build-script/issues/21)）

