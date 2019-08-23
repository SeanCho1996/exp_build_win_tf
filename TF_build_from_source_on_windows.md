# 系统环境
- Windows 10
- Anaconda: 4.7.10
- Python: 3.6.7
- CUDA: 10.0
- CUDNN: 7.6.1
- Tensorflow: 1.13.1 gpu


# 预安装编译环境
1. 在anaconda powershell prompt (以下简称cmd)中创建新的虚拟环境tf_test

```
conda create –n tf_test python=3.6.7
conda activate tf_test
```

2. cmd下安装以下Tensorflow pip软件包依赖项

```
pip install six numpy wheel
pip install keras_applications==1.0.6 --no-deps
pip install keras_preprocessing==1.0.5 --no-deps
```

3. 安装[Bazel](https://github.com/bazelbuild/bazel/releases)作为Tensorflow的编译工具，对于将要安装的Tensorflow_gpu-1.13.1，此流程经测试确认可用的Bazel版本为0.21.0，将安装好的bazel.exe添加到系统环境变量%PATH%中

4. 安装[MSYS2](https://www.msys2.org)作为Tensorflow所需的bin工具，默认的安装路径为C:/msys64，将C:\msys64\usr\bin添加到%PATH%环境变量中。然后，在cmd下运行以下命令。(*下载很慢，容易断掉，如何处理？用手机热点，多试几次*)

```
pacman –S git patch unzip
```

5. 安装[Visual studio C++生成工具](https://visualstudio.microsoft.com/zh-hans/vs/older-downloads)，虽然生成工具可以独立安装，但这里建议安装完整的Visual studio 2015（对于将要安装的Tensorflow_gpu-1.13.1，此流程经测试确认可用的VS版本为VS 2015），安装时注意勾选通用 **Windows应用开发工具**->**Tools和Windows10 SDK** 以及 **通用工具**->**Visual Studio扩展性工具Update3**

6. 4.7及以上版本的Anaconda安装时可以提供`powershell`提示符，可以直接运行`ps1`文件，无需额外设置。


# 编译过程
1. 在powershell中进入源码路径，路径下应有以下文件或文件夹：
文件夹：patches（用于C++ API编译的补丁包），source（tensorflow源代码）
文件：build.ps1（编译脚本），extract-built-tensorflow-cpp-api.ps1（lib和include路径生成脚本）
```
(tf_test) (base) PS C:\Users\Administrator>cd C:\build_tf
```

2. 在当前路径下运行脚本build.ps1

	需激活的参数
- `BuildCppAPI`：编译C++ API时需激活此参数，用于应用C++ API补丁以及调用外部符号
- `ReserveSource`：保留之前编译的结果
- `IgnoreDepsVersionIssues`：编译过程中忽略编译工具版本不对应
- `InstallDefaultDeps`：安装默认版本的编译工具

	需输入的参数
- `BazelBuildParameter`：bazel编译参数，添加在 " " 内，细节参阅 [tensorflow官方文档](https://tensorflow.google.cn/install/source_windows#gpu_support)

	```
	(tf_test) (base) PS C:\build_tf>.\build.ps1 –BuildCppAPI –BazelBuildParameter “--config=opt --config=cuda --define=no_tensorflow_py_deps=true //tensorflow:libtensorflow_cc.so”
	```

	- 以上指令将编译C++ API的libtensorflow_cc.so
	- 编译初始阶段会从github上下载多个第三方文件，需要能够翻墙的网络（挂载vpn后，移动网速较联通更快）
	- 下载失败的package可以反复尝试，已下载的package无需重复下载
	- 该脚本主要修改自 [`build.ps1`](https://github.com/guikarist/tensorflow-windows-build-script/blob/master/build.ps1)，可定期同步更新

3. 经过较长时间的编译后，powershell将输出

	```
	INFO: Build completed successfully
	```
	至此Tensorflow库文件编译完成，输出路径为`C:\build_tf\source\bazel-out\x64_windows-opt\bin\tensorflow\libtensorflow_cc.so`

4. 若需要生成tensorflow_cc.dll, tensorflow_cc.lib及include文件，在当前目录下运行extract-built-tensorflow-cpp-api.ps1，输出的库文件和头文件在新生成的bin\tensorflow文件夹中
	```
	(tf_test) (base) PS C:\build_tf> .\extract-built-tensorflow-cpp-api.ps1
	```

