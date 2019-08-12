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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include <Eigen\Dense>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>
#include <stdio.h>

using namespace tensorflow;  // NOLINT(build/namespaces)
using namespace Eigen;
using namespace std;
using Tensor2i = tensorflow::TTypes<int32, 2>;
using Tensor2f = tensorflow::TTypes<float, 2>;

__global__ void AddOneKernel(const Tensor2i::ConstTensor in, Tensor2f::Tensor out) {

	int layers = in.dimensions()[0];
	int length = in.dimensions()[1];

	// Convert input, output Eigen::TensorMap to Eigen::Map(Matrix) of 4*1000
	Eigen::Map<const Eigen::MatrixXi> in_map(in.data(), length, layers);
	Eigen::Map<Eigen::MatrixXf> out_map(out.data(), length, layers);

	// Calculate inverse for each layer
	for (int i = 0; i < layers; i++) {
		Eigen::Vector4i in_vec1 = in_map.col(i);

		Eigen::Map<Eigen::Matrix2i> m2(in_vec1.data()); // reshape to 2*2 for calculation
		Eigen::MatrixXf m2f = m2.cast<float>().inverse(); // convert dtype and calculate inverse

		Eigen::Map<Eigen::VectorXf> m2col(m2f.data(), 4, 1); // reshape back to col for assignment
		out_map.col(i) = m2col;
	}
}

void AddOneKernelLauncher(tensorflow::OpKernelContext* context, const Tensor2i::ConstTensor &in, Tensor2f::Tensor &out) {
	AddOneKernel<<<1, 1>>>(in, out);
}

__global__ void AddOneKernel2(const TTypes<int32>::ConstFlat in, Tensor2i::Tensor out) {

	auto aaa = in.data();
	printf("%d\n", aaa[0]);
	//printf("%d\n", aaa[1]);

}
void AddOneKernelLauncher2(const TTypes<int32>::ConstFlat &in, Tensor2i::Tensor &out)
{
	AddOneKernel2 << <32, 256 >> >(in, out);
}

#endif
