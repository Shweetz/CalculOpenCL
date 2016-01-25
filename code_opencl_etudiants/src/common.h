#include <cstdio>
#include <cstdlib>

#define uint  unsigned int


/**
  Classical matrix structure.
*/
typedef struct matrix
{
	uint w; // width
	uint h; // height
	float *data; // pointer to data
} Matrix;


/**
  CSR matrix structure.
*/
typedef struct matrixCSR
{
	uint w; // width
	uint h; // height
	uint nzNbr; // number of non-zero values
	float *data; // array of non zero values
	uint *col_ind; // array of column index
	uint *row_ptr; // array of pointers to rows
} MatrixCSR;


/**
  ELL matrix structure.
*/
typedef struct matrixELL
{
	uint w; // width
	uint h; // height
	uint nzRowSz; // max number of non zero values in a row
	float *data; // array of non zero values
	uint *col_ind; // array of column index
} MatrixELL;


/**
  Create a matrix structure. Allocate memory for data.
  Memory must be deallocated by user iby calling deleteMatrix().
*/
Matrix* createMatrix( uint w, uint h);


/**
  Destroy a matrix structure.
*/
void deleteMatrix(Matrix **m);


/**
  Write square matrix to file.
*/
void writeMatrixToFile( const float *data, uint width, uint height, const char *fileName);
void writeMatrixToFile( const Matrix *m, const char *fileName);


/**
  Read square matrix from file.
*/
Matrix* readMatrixFromFile(const char *fileName);


/**
  Initialize a matrix with random values.
  @param  width  matrix width
  @param  height  matrix height
  @param  matrix  pointer to allocated memory area
  @param  sparseRate  rate of zero values in [0.0;99.9]
*/
void initMatrix( uint width, uint height, float *data, float sparseRate);
void initMatrix( const Matrix *m, float sparseRate);


/**
  Compare 2 matrices. Returns 'true' if they are equal.
*/
bool areEqual( uint width, uint height, const float *data1, const float *data2);
bool areEqual( const Matrix *m1, const Matrix *m2);

/**
  Print matrix to stdout.
*/
void printMatrix( const Matrix *m, const char *title);

/**
  Convert a classical Matrix to a MatrixCSR.
*/
MatrixCSR* matrixToCSR(const Matrix *m);

/**
  Destroy a matrix structure.
*/
void deleteMatrixCSR(MatrixCSR **m);

/**
  Print CSR matrix to stdout.
*/
void printMatrixCSR( const MatrixCSR *m, const char *title);


/**
  Convert a classical Matrix to a MatrixELL.
*/
MatrixELL* matrixToELL(const Matrix *m);

/**
  Destroy a matrix structure.
*/
void deleteMatrixELL(MatrixELL **m);

/**
  Print ELL matrix to stdout.
*/
void printMatrixELL( const MatrixELL *m, const char *title);


/**
  Do matrix-vector multiplication with various methods.
*/
bool checkResult(const char *title, const Matrix *reference, const Matrix *result);




#if 0
#ifndef CUDA
	/**
	  Do computation on GPU.
	*/
	int doGpuComputation( int matrixWidth, const char *kernelName, int dimension, size_t *globalWorkSize, size_t *workGroupSize);
#else
	// CUDA debugging
	#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
	#define gpuKernelExecErrChk() { gpuCheckKernelExecutionError( __FILE__, __LINE__); }

	inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false)
	{
		if (code != cudaSuccess) 
		{
			fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
			if( abort )
				exit(code);
		}
	}

	inline void gpuCheckKernelExecutionError( const char *file, int line)
	{
		/**
		  Check for invalid launch argument, then force the host to wait 
		  until the kernel stops and checks for an execution error.
		  The synchronisation can be eliminated if there is a subsequent blocking
		  API call like cudaMemcopy. In this case the cudaMemcpy call can return
		  either errors which occurred during the kernel execution or those from 
		  the memory copy itself. This can be confusing for the beginner, so it is
		  recommended to use explicit synchronisation after a kernel launch during
		  debugging to make it easier to understand where problems might be arising. 
		 */
		gpuAssert( cudaPeekAtLastError(), file, line);
		gpuAssert( cudaDeviceSynchronize(), file, line);	
	}
#endif
#endif
