/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "third_party\eigen3\unsupported\Eigen\CXX11\Tensor"
#include <Eigen\Dense>
#include <iostream>

using namespace tensorflow;  // NOLINT(build/namespaces)
using namespace Eigen;
using Tensor2i = tensorflow::TTypes<int32, 2>;
using Tensor2f = tensorflow::TTypes<float, 2>;
using namespace std;

REGISTER_OP("AddOne")
    .Input("input: int32")
    .Output("output: float")
    .Doc(R"doc(
Adds 1 to all elements of the tensor.

output: A Tensor.
  output = input + 1
)doc");

//void AddOneKernelLauncher(const int* in, const int N, int* out);
void AddOneKernelLauncher(tensorflow::OpKernelContext* context, const Tensor2i::ConstTensor &in, Tensor2f::Tensor &out);
void AddOneKernelLauncher2(const TTypes<int32>::ConstFlat &in, Tensor2i::Tensor &out);

class AddOneOp : public OpKernel {
 public:
  explicit AddOneOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const tensorflow::Tensor& input_tensor = context->input(0);
    // auto input = input_tensor.flat<int32>();


	// Convert input tensor to Eigen::Tensor
	auto in = input_tensor.flat_outer_dims<int32, 2>();

    // Create an output tensor
	tensorflow::Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));

	// Convert output tensor to Eigen::Tensor
	auto out = output_tensor->flat_outer_dims<float, 2>();
	//auto out = output_tensor->shaped<int32, 3>({ 1000, 2, 2 });


    // Call the cuda kernel launcher
    AddOneKernelLauncher(context, in, out);

  }
};

REGISTER_KERNEL_BUILDER(Name("AddOne").Device(DEVICE_GPU), AddOneOp);
