#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "third_party\eigen3\unsupported\Eigen\CXX11\Tensor"
#include <Eigen\Dense>
#include <iostream>

using namespace tensorflow;  // NOLINT(build/namespaces)
using namespace Eigen;
using Tensor2c = tensorflow::TTypes<complex64, 2>;
using Tensor1f = tensorflow::TTypes<float, 1>;
using Tensor2i = tensorflow::TTypes<int16, 2>;
using namespace std;

typedef Eigen::Matrix<int16, -1, -1> MatrixXi16;

REGISTER_OP("CsmToRcm")
.Input("csm: complex64")
.Input("reg: float")
.Input("idx0: int16")
.Input("idx1: int16")
.Input("idx2: int16")
.Input("idx3: int16")
.Output("rcm: complex64");

//void AddOneKernelLauncher(const int* in, const int N, int* out);
void CsmToRcmKernelLauncher_PtType4_parrel(Eigen::Map<const Eigen::MatrixXcf> &csm_map, Eigen::Map<const Eigen::VectorXf> &reg_vec,
	Eigen::Map<const MatrixXi16> &offset8,		// (12160, 8, 3)
	Eigen::Map<const MatrixXi16> &offset3,		// (9728, 3, 3)
	Eigen::Map<const MatrixXi16> &offset4,		// (12160, 4, 3)
	Eigen::Map<const MatrixXi16> &offset6,		// (9728, 6, 3)
	Eigen::Map<Eigen::MatrixXcf> &rcm_map);
//void AddOneKernelLauncher2(const TTypes<int32>::ConstFlat &in, Tensor2i::Tensor &out);

class CsmToRcmOp : public OpKernel {
public:
	explicit CsmToRcmOp(OpKernelConstruction* context) : OpKernel(context) {}

	// realize sense using native float type
	void Compute(OpKernelContext* context) override
	{
		auto t1 = std::chrono::system_clock::now();

		// Grab the input tensor
		const tensorflow::Tensor& csm_tensor = context->input(0);
		const tensorflow::Tensor& reg_tensor = context->input(1);

		const tensorflow::Tensor& idx0_tensor = context->input(2);
		const tensorflow::Tensor& idx1_tensor = context->input(3);
		const tensorflow::Tensor& idx2_tensor = context->input(4);
		const tensorflow::Tensor& idx3_tensor = context->input(5);

		// Create an output tensor
		tensorflow::Tensor* rcm_tensor = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output(0, csm_tensor.shape(), &rcm_tensor));

		//auto csm = csm_tensor.flat<complex64>();
		//auto reg = reg_tensor.flat<float>();
		//auto rcm = rcm_tensor->flat<complex64>();
		
		// Convert tensor to Eigen::tensormap and compress dimensions
		auto csm = csm_tensor.flat_inner_dims<complex64, 2>();
		auto reg = reg_tensor.flat_inner_dims<float, 1>();
		auto rcm = rcm_tensor->flat_inner_dims<complex64, 2>();
		

		//auto idx0 = idx0_tensor.flat<int16>();
		//const int N0 = idx0_tensor.shape().dim_size(0);
		//std::cout << "N0 = " << N0 << std::endl;
		//int ptsCnt = idx0_tensor.shape().dim_size(1);
		//int idxCnt = idx0_tensor.shape().dim_size(2);
		auto idx0 = idx0_tensor.flat_inner_dims<int16, 2>();
		auto idx1 = idx1_tensor.flat_inner_dims<int16, 2>();
		auto idx2 = idx2_tensor.flat_inner_dims<int16, 2>();
		auto idx3 = idx3_tensor.flat_inner_dims<int16, 2>();

		int dim0 = idx0_tensor.dims();
		int dim1 = idx1_tensor.dims();
		int dim2 = idx2_tensor.dims();
		int dim3 = idx3_tensor.dims();

		int layer = csm.dimensions()[0];
		int length = csm.dimensions()[1];
		
		// Map tensormap to Eigen::Matrix
		Eigen::Map<const Eigen::MatrixXcf> csm_map(csm.data(), length, layer); // input matrix (16 * 233472)
		Eigen::Map<Eigen::MatrixXcf> rcm_map(rcm.data(), length, layer); // ouput matrix (16 * 233472)

		Eigen::Map<const Eigen::VectorXf> reg_vec(reg.data(), length, 1); //coeff vector (1 * 233472)
		Eigen::Map<const MatrixXi16> offset8(idx0.data(), idx0.dimensions()[1], idx0.dimensions()[0]); // 8-point offset matrix (3 * 97280)
		Eigen::Map<const MatrixXi16> offset3(idx1.data(), idx1.dimensions()[1], idx1.dimensions()[0]); // 3-point offset matrix (3 * 29184)
		Eigen::Map<const MatrixXi16> offset4(idx2.data(), idx2.dimensions()[1], idx2.dimensions()[0]); // 4-point offset matrix (3 * 48640)
		Eigen::Map<const MatrixXi16> offset6(idx3.data(), idx3.dimensions()[1], idx3.dimensions()[0]); // 6-point offset matrix (3 * 58368)

		// ONLY test 2 x 2 acc
		if (idx1_tensor.dims() == 0)
			CsmToRcmKernelLauncher_PtType4_parrel(csm_map, reg_vec, offset8, offset3, offset4, offset6, rcm_map);

		///*
		// ONLY test 3.6 x 1.5 acc
		if (dim0 > 0 && dim1 > 0 && dim2 > 0 && dim3 > 0)
		{
			CsmToRcmKernelLauncher_PtType4_parrel(csm_map, reg_vec,
				offset8,		// (12160, 8, 3)
				offset3,		// (9728, 3, 3)
				offset4,		// (12160, 4, 3)
				offset6,		// (9728, 6, 3)
				rcm_map);
		}
		//*/

		/*

		// create handle cost nearly 200ms
		cublasHandle_t handle;
		cublasCreate(&handle);

		calcRcmKernelLauncher_Serial(handle, idx0_tensor, csm, reg, rcm);
		calcRcmKernelLauncher_Serial(handle, idx1_tensor, csm, reg, rcm);
		calcRcmKernelLauncher_Serial(handle, idx2_tensor, csm, reg, rcm);
		calcRcmKernelLauncher_Serial(handle, idx3_tensor, csm, reg, rcm);

		cublasDestroy(handle);

		*/

		auto t2 = std::chrono::system_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		std::cout << "CsmToRcmKernelLauncher_PtType4 time consumed: " << duration * 1e-3 << std::endl;

	}


	/*
	// try to realize sense using eigen::tensor
	void Compute(OpKernelContext* context) override
	{
	// Grab the input tensor
	const Tensor& csm_tensor = context->input(0);
	const Tensor& reg_tensor = context->input(1);
	const Tensor& idx_tensor = context->input(2);

	// Create an output tensor
	Tensor* rcm_tensor = nullptr;
	OP_REQUIRES_OK(context, context->allocate_output(0, csm_tensor.shape(), &rcm_tensor));

	// csm is Eigen::Tensor
	//auto csm = csm_tensor.flat_outer_dims<complex64, 3>();
	//int dim0 = csm.dimensions()[0];
	//int dim1 = csm.dimensions()[1];
	//int dim2 = csm.dimensions()[2];
	//std::cout << "dims: " << dim0 << std::endl;
	//std::cout << "dims: " << dim1 << std::endl;
	//std::cout << "dims: " << dim2 << std::endl;

	auto csm = csm_tensor.tensor<complex64, 4>();
	auto reg = reg_tensor.tensor<float, 3>();
	auto idx = idx_tensor.tensor<int16, 3>();
	auto rcm = rcm_tensor->tensor<complex64, 4>();

	CsmToRcmKernelLauncherTensor(csm, reg, idx, rcm);
	}
	*/
};

REGISTER_KERNEL_BUILDER(Name("CsmToRcm").Device(DEVICE_GPU), CsmToRcmOp);