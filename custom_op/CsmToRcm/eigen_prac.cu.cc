#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include <Eigen\Dense>
#include <Eigen/LU>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <stdio.h>
#include <math.h>
#include <complex>


using namespace tensorflow;  // NOLINT(build/namespaces)
using namespace Eigen;
using Tensor2c = tensorflow::TTypes<complex64, 2>;
using Tensor1f = tensorflow::TTypes<float, 1>;
using Tensor2i = tensorflow::TTypes<int16, 2>;
using namespace std;

typedef Eigen::Matrix<int16, -1, -1> MatrixXi16;
typedef Eigen::Matrix<float, 8, 8> Matrix8f;



__global__ void CsmToRcmKernel(Eigen::Map<const Eigen::MatrixXcf> &csm_map, Eigen::Map<const Eigen::VectorXf> &reg_vec,
	Eigen::Map<const MatrixXi16> &offset8,		// (12160, 8, 3)
	Eigen::Map<const MatrixXi16> &offset3,		// (9728, 3, 3)
	Eigen::Map<const MatrixXi16> &offset4,		// (12160, 4, 3)
	Eigen::Map<const MatrixXi16> &offset6,		// (9728, 6, 3)
	Eigen::Map<Eigen::MatrixXcf> &rcm_map) {

	int i_t = blockIdx.x * blockDim.x + threadIdx.x;
	int nSL = 48;
	int nFE = 76;
	int nCH = 16;

	// 4-point condition
	// Extract target vectors and their coeffcient
	Eigen::MatrixXcf csm_tmp(16, 4);
	Eigen::VectorXf r_vec(4);
	int csm_idx[4];
	for (int j = 0; j < 4; j++) {
		Eigen::Vector3i idx_vec = offset8.col(i_t*4 + j).cast<int32>();
		csm_idx[j] = idx_vec(0) * nSL * nFE  +
				     idx_vec(1) * nFE  +
				     idx_vec(2) ;
		csm_tmp.col(csm_idx[j]) = csm_map.col(csm_idx[j]); // a single point
		r_vec(j) = reg_vec(csm_idx[j]); // coeff for the point	
		
	}


	// SH = conj(squeeze(senseMap(wPE, wSL, FE, :))); 
	Eigen::MatrixXcf csm_tmp_t = csm_tmp.transpose(); // 16*4 -> 4*16, correspond to SH
	csm_tmp_t = csm_tmp_t.conjugate();

	// R = diag(reshape(RegI(wPE,wSL,FE),Rf,1));
	Eigen::MatrixXf R_tmp(r_vec.asDiagonal());

	// A = B*SH' + R;
	Eigen::Matrix4f A = (csm_tmp_t * csm_tmp).real() + R_tmp;

	// inv_A = inv(A);
	Eigen::Matrix4f inv_A = A.inverse();
	//Eigen::Map<const Matrix8f> inv_A(A.inverse(), 8, 8);
	//printf("%f", inv_A(0, 0));
	//printf("%f abcd \n", B(0, 0));
	//printf("%d ", A.inverse().rows());
	//printf("%d \n", A.inverse().cols());

	// C(ind,:) = inv_A*B;
	Eigen::Matrix4cf inv_Ac;
	inv_Ac.real() = inv_A;
	inv_Ac.imag() = Eigen::Matrix4f::Zero(4,4);
	Eigen::Matrix4cf C = inv_Ac * csm_tmp_t;

	// g_tmp(ind) = sqrt(diag(inv_A).*diag(A));
	float g_tmp = inv_A.diagonal().dot(A.diagonal());
	g_tmp = sqrt(g_tmp);

	// Prepare to export
	Eigen::MatrixXcf C_t = C.transpose(); // return to 16*4
	for (int j = 0; j < 4; j++) {
		 rcm_map.col(csm_idx[j]) = csm_tmp.col(j);
		if (i_t == 4095)
			printf("%d \n", csm_idx[j]);
	}
	//printf("%d \n", i_t);
	
}

void CsmToRcmKernelLauncher_PtType4_parrel(Eigen::Map<const Eigen::MatrixXcf> &csm_map, Eigen::Map<const Eigen::VectorXf> &reg_vec,
	Eigen::Map<const MatrixXi16> &offset8,		// (12160, 8, 3)
	Eigen::Map<const MatrixXi16> &offset3,		// (9728, 3, 3)
	Eigen::Map<const MatrixXi16> &offset4,		// (12160, 4, 3)
	Eigen::Map<const MatrixXi16> &offset6,		// (9728, 6, 3)
	Eigen::Map<Eigen::MatrixXcf> &rcm_map){
	auto t1 = std::chrono::system_clock::now();
	CsmToRcmKernel <<<1,256 >>>(csm_map, reg_vec, offset8, offset3, offset4, offset6, rcm_map);
	auto t2 = std::chrono::system_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
	printf("inverse time consumed: %f", duration * 1e-3);
}



#endif
