#include<stdio.h>
#include<stdexcept>

#include"tools.h"
#include"common.h"
#include"mult_mat_vect_opencl.h"


/**
  Compute M1xM2 on CPU. Classical method.
*/
Matrix* cpuSpmvClassical(const Matrix *m1, const Matrix *m2)
{
	if(m1->w != m2->h)
		throw std::runtime_error("Failed to multiply matrices, size mismatch.");

	// output matrix size
	uint width = m2->w;
	uint height = m1->h;
	Matrix *m1xm2 = createMatrix(width, height);

	top(0);
	for(uint r = 0; r < height; r++)
	{
		for(uint c = 0; c < width; c++)
		{
			float tmp = 0;

			for(uint k = 0; k < m1->w; k++)
				tmp += m1->data[r*(m1->w) + k] * m2->data[k*(m2->w) + c];

			m1xm2->data[r*width + c] = tmp;
		}
	}
	double cpuRunTime = top(0);

	printf("Classical method on cpu: M(%dx%d)xV computed in %f ms.\n", m1->w, m1->h, cpuRunTime);

	return m1xm2;
}


/**
  Do matrix-vector multiplication with various methods.
*/
int main(int argc, const char **argv)
{
	if(argc != 2)
	{
		printf("Usage: %s dataset_basename\n", argv[0]);
		printf("Example: %s  mat_1000x1500_0.50\n", argv[0]);
		return 1;
	}

	std::string matrixFileName = std::string(argv[1]) + ".M";
	std::string vectorFileName = std::string(argv[1]) + ".V";

	Matrix *m = readMatrixFromFile(matrixFileName.c_str());
	Matrix *v = readMatrixFromFile(vectorFileName.c_str());

	// classical method on CPU, used as reference
	Matrix *mv_cpu_classical = cpuSpmvClassical(m, v);

	// CSR method on GPU
	MatrixCSR *mCSR = matrixToCSR(m);
	Matrix *mv_gpu_csr = gpuSpmvCSR(mCSR, v, mv_cpu_classical);
	deleteMatrix(&mv_gpu_csr);

	// CSR-Vect method on GPU
	Matrix *mv_gpu_csr_vect = gpuSpmvCSRVect(mCSR, v, mv_cpu_classical);
	deleteMatrix(&mv_gpu_csr_vect);

	// release memory
	deleteMatrix(&m);
	deleteMatrix(&v);
	deleteMatrix(&mv_cpu_classical);
	deleteMatrixCSR(&mCSR);

	return 0;
}

