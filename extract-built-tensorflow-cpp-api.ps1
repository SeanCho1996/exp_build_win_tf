# This script should be executed outside repo folder of https://github.com/guikarist/tensorflow-windows-build-script.
Set-StrictMode -Version latest
$ErrorActionPreference = "Stop"

Remove-Item bin -ErrorAction SilentlyContinue -Force -Recurse

$tensorFlowBinDir = "$pwd\bin"
mkdir $tensorFlowBinDir | Out-Null

$tensorFlowSourceDir = "$pwd\tensorflow-windows-build-script\source"

$rootdir = "$pwd"

# Tensorflow lib and includes
mkdir "$tensorFlowBinDir/tensorflow/lib" | Out-Null
Copy-Item  $rootdir\source\bazel-bin\tensorflow\libtensorflow_cc.so $tensorFlowBinDir\tensorflow\lib\tensorflow_cc.dll
Copy-Item  $rootdir\source\bazel-bin\tensorflow\liblibtensorflow_cc.so.ifso $tensorFlowBinDir\tensorflow\lib\tensorflow_cc.lib

Copy-Item $rootdir\source\tensorflow\core $tensorFlowBinDir\tensorflow\include\tensorflow\core -Recurse -Container  -Filter "*.h"
Copy-Item $rootdir\source\tensorflow\cc $tensorFlowBinDir\tensorflow\include\tensorflow\cc -Recurse -Container -Filter "*.h"
Copy-Item $rootdir\source\bazel-genfiles\tensorflow\cc\ops\*.h  $tensorFlowBinDir\tensorflow\include\tensorflow\cc\ops
Copy-Item $rootdir\source\bazel-genfiles\tensorflow\core\framework\*.h $tensorFlowBinDir\tensorflow\include\tensorflow\core\framework