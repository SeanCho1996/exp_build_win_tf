load("//tensorflow:tensorflow.bzl", "tf_custom_op_library", "tf_gen_op_libs", "tf_kernel_library",)
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured", "if_cuda")

tf_custom_op_library(
    name = "AddOne.dll",
    srcs = ["eigen_4d.cc"],
	gpu_srcs = ["eigen_4d.cu.cc"],
)