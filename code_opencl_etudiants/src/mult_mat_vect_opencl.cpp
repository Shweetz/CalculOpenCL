#define __CL_ENABLE_EXCEPTIONS
#include"cl.hpp"

#include<cstdio>
#include<cstdlib>
#include<iostream>
#include<stdexcept>
#include<math.h>  // for ceil()

#include"tools.h"
#include"common.h"


//---------------------------------------------------------

// STUDENTS BEGIN

std::string kernelSpmvCSR_source =
"__kernel__ void spmvCSR(uint rowsNbr, const __global float *values, const __global uint *col_ind, const __global uint *row_ptr, const __global float *v, __global float *y)
{\n\
	uint r = get_global_id(0);\n\
	if( r < rowsNbr )\n\
	{\n\
		float dot = 0.0f;\n\
		int row_beg = row_ptr[r];\n\
		int row_end = row_ptr[r+1];\n\
\n\
		for(uint i = row_beg; i < row_end; i++)\n\
			dot += values[i] * v[col_ind[i]];\n\
\n\
		y[r] = dot;\n\
	}\n\
}\n\
";

// STUDENTS END

//---------------------------------------------------------

/**
  Compute MxV on GPU. CSR method.
  A reference result can be passed to check that the computation is ok.
*/
Matrix* gpuSpmvCSR(const MatrixCSR *m, const Matrix *v, const Matrix *reference)
{
	const char *name = "CSR method on GPU";

	if(m->w != v->h)
		throw std::runtime_error("Failed to multiply matrices, size mismatch.");
	if(v->w != 1)
		throw std::runtime_error("Failed to multiply matrices, vector size mismatch.");

	// output matrix size
	uint width = v->w;
	uint height = m->h;
	Matrix *mv = createMatrix(width, height);

	// time measurement storage
	double gpuRunTime = 0;
	double gpuComputeTime = 0;

	try
	{
		// init OpenCL

		// retreive list of available platforms and select the first one
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		if( platforms.size() == 0 )
			throw std::runtime_error("No OpenCL platform found. Check installation!\n");
		cl::Platform platform = platforms[0];
		std::cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << "\n";

		// get the first GPU device of the selected platform
		std::vector<cl::Device> devices;
		platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
		cl::Device device = devices[0];

		// display informations on device
		std::cout << "Using device:\n";
		std::cout << "  CL_DEVICE_NAME    = " << device.getInfo<CL_DEVICE_NAME>() << "\n";
		std::cout << "  CL_DEVICE_VENDOR  = " << device.getInfo<CL_DEVICE_VENDOR>() << "\n";
		std::cout << "  CL_DEVICE_VERSION = " << device.getInfo<CL_DEVICE_VERSION>() << "\n";
		std::cout << "  CL_DRIVER_VERSION = " << device.getInfo<CL_DRIVER_VERSION>() << "\n";

		// create a context with the GPU device
		cl::Context context(devices);

		// create command queue using the context and device
		cl::CommandQueue queue(context, device);
		
		// init ok
		printf("Compute device successfully initialized.\n");

		// compile kernel

		// create a program from the kernel source code
		cl::Program::Sources sources;
		sources.push_back(std::make_pair(kernelSpmvCSR_source.c_str(), kernelSpmvCSR_source.length()));
		cl::Program program(context, sources);

		// compile program
		try
		{
			program.build(devices);
		}
		catch( cl::Error err )
		{
			// display build log
			std::cout << "Program build log:\n";
			std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";

			throw; // propagate exception
		}
		printf("Program successfully built.\n");

		// specify which kernel to execute
		// (the program may contain several kernels)
		cl::Kernel kernel(program, "kernelSpmvCSR");

		// allocate global memory on GPU
		uint valuesSizeInBytes = m->nzNbr * sizeof(float);
		uint col_indSizeInBytes = m->nzNbr * sizeof(uint);
		uint row_ptrSizeInBytes = (m->h + 1) * sizeof(uint);
		uint vSizeInBytes = (v->h) * sizeof(float);
		uint mvSizeInBytes = (m->h) * sizeof(float);
		cl::Buffer gpuValues(context, CL_MEM_READ_ONLY, valuesSizeInBytes);
		cl::Buffer gpuCol_ind(context, CL_MEM_READ_ONLY, col_indSizeInBytes);
		cl::Buffer gpuRow_ptr(context, CL_MEM_READ_ONLY, row_ptrSizeInBytes);
		cl::Buffer gpuV(context, CL_MEM_READ_ONLY, vSizeInBytes);
		cl::Buffer gpuMV(context, CL_MEM_WRITE_ONLY, mvSizeInBytes); // result of matrix-vect multiplication

		// transfer data from CPU memory to GPU memory
		top(0); // start time measurement
		queue.enqueueWriteBuffer(gpuValues, CL_TRUE, 0, valuesSizeInBytes, m->data);
		queue.enqueueWriteBuffer(gpuCol_ind, CL_TRUE, 0, col_indSizeInBytes, m->col_ind);
		queue.enqueueWriteBuffer(gpuRow_ptr, CL_TRUE, 0, row_ptrSizeInBytes, m->row_ptr);
		queue.enqueueWriteBuffer(gpuV, CL_TRUE, 0, vSizeInBytes, v->data);
		queue.enqueueWriteBuffer(gpuMV, CL_TRUE, 0, mvSizeInBytes, mv->data);

		// STUDENTS BEGIN

		// Set the arguments to our compute kernel
		//TODO
		//kernel.setArg(0, m->h);

		// run kernel
		top(1);
		//TODO

		// STUDENTS END

		// Wait for the command queue to get serviced before reading back results
		queue.finish();
		gpuComputeTime = top(1); // pure computation duration

		// transfer data from GPU memory to CPU memory
		queue.enqueueReadBuffer(gpuMV, CL_TRUE, 0, mvSizeInBytes, mv->data);
		gpuRunTime = top(0); // computation and memory transfert duration
	}
	catch( cl::Error err )
	{
		std::cerr
			<< "ERROR: "
			<< err.what()
			<< "("
			<< err.err()
			<< ")"
			<< std::endl;

		if( err.what() == std::string("clBuildProgram") )
		{
			const char* error_type;
		
			if( err.err() == CL_INVALID_PROGRAM )
				error_type = "CL_INVALID_PROGRAM";
			else if( err.err() == CL_INVALID_VALUE )
				error_type = "CL_INVALID_VALUE";
			else if( err.err() == CL_INVALID_DEVICE )
				error_type = "CL_INVALID_DEVICE";
			else if( err.err() == CL_INVALID_BINARY )
				error_type = "CL_INVALID_BINARY";
			else if( err.err() == CL_INVALID_BUILD_OPTIONS )
				error_type = "CL_INVALID_BUILD_OPTIONS";
			else if( err.err() == CL_INVALID_OPERATION )
				error_type = "CL_INVALID_OPERATION";
			else if( err.err() ==  CL_COMPILER_NOT_AVAILABLE )
				error_type = " CL_COMPILER_NOT_AVAILABLE";
			else if( err.err() == CL_BUILD_PROGRAM_FAILURE )
				error_type = "CL_BUILD_PROGRAM_FAILURE";
			else if( err.err() == CL_INVALID_OPERATION )
				error_type = "CL_INVALID_OPERATION";
			else if( err.err() == CL_OUT_OF_HOST_MEMORY )
				error_type = "CL_OUT_OF_HOST_MEMORY";
		
			printf( "Program build error: %s.\n", error_type);
		}

		throw std::runtime_error("Aborting.");
	}

	// check result, display run time if result is correct
	bool displayRunTime = true;
	if( reference )
	{
		if(! checkResult(name, reference, mv))
			displayRunTime = false;
	}

	if(displayRunTime)
		printf("%s: M(%dx%d)xV computed in %f ms (%f ms of pure computation).\n", name, m->w, m->h, gpuRunTime, gpuComputeTime);

	return mv;
}

//---------------------------------------------------------

// STUDENTS BEGIN

std::string kernelSpmvCSRVect_source =
"__kernel__ void kernelSpmvCSRVect(uint rowsNbr, const __global float *values, const __global uint *col_ind, const __global uint *row_ptr, 
const __global float *v, __global float *y, __local float *dots)
{
	// dots is dynamically allocated in local memory, size given as kernel arg
	
	uint threadId = get_global_id(0); // global thread index
	uint localId = get_local_id(0); // thread index in workgroup
	uint warpId = threadId / 32; // global warp index
	uint lane = threadId % 32; // thread index within the warp

	uint r = warpId; // one row per warp

	if( r < rowsNbr )
	{
		int row_beg = row_ptr[r];
		int row_end = row_ptr[r+1];
		dots[localId] = 0.0f;

		for(uint i = row_beg + lane; i < row_end; i+=32)
			dots[localId] += values[i] * v[col_ind[i]];

		// parallel reduction in shared memory
		if( lane < 16 )  dots[localId] += dots[localId + 16];
		if( lane <  8 )  dots[localId] += dots[localId +  8];
		if( lane <  4 )  dots[localId] += dots[localId +  4];
		if( lane <  2 )  dots[localId] += dots[localId +  2];
		if( lane <  1 )  dots[localId] += dots[localId +  1];

		// first thread writes the result in global memory
		if( lane == 0 )
			y[r] = dots[localId];
	}
}
";

// STUDENTS END

//---------------------------------------------------------

/**
  Compute MxV on GPU. CSR-Vect method.
  A reference result can be passed to check that the computation is ok.
*/
Matrix* gpuSpmvCSRVect(const MatrixCSR *m, const Matrix *v, const Matrix *reference)
{
	const char *name = "CSR-Vect method on GPU";

	if(m->w != v->h)
		throw std::runtime_error("Failed to multiply matrices, size mismatch.");
	if(v->w != 1)
		throw std::runtime_error("Failed to multiply matrices, vector size mismatch.");

	// output matrix size
	uint width = v->w;
	uint height = m->h;
	Matrix *mv = createMatrix(width, height);

	// time measurement storage
	double gpuRunTime = 0;
	double gpuComputeTime = 0;

#if 0
	try
	{
		// init OpenCL

		// retreive list of available platforms and select the first one
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		if( platforms.size() == 0 )
			throw std::runtime_error("No OpenCL platform found. Check installation!\n");
		cl::Platform platform = platforms[0];
		std::cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << "\n";

		// get the first GPU device of the selected platform
		std::vector<cl::Device> devices;
		platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
		cl::Device device = devices[0];

		// display informations on device
		std::cout << "Using device:\n";
		std::cout << "  CL_DEVICE_NAME    = " << device.getInfo<CL_DEVICE_NAME>() << "\n";
		std::cout << "  CL_DEVICE_VENDOR  = " << device.getInfo<CL_DEVICE_VENDOR>() << "\n";
		std::cout << "  CL_DEVICE_VERSION = " << device.getInfo<CL_DEVICE_VERSION>() << "\n";
		std::cout << "  CL_DRIVER_VERSION = " << device.getInfo<CL_DRIVER_VERSION>() << "\n";

		// create a context with the GPU device
		cl::Context context(devices);

		// create command queue using the context and device
		cl::CommandQueue queue(context, device);
		
		// init ok
		printf("Compute device successfully initialized.\n");

		// compile kernel

		// STUDENTS BEGIN

		// create a program from the kernel source code
		
		// compile program

		// specify which kernel to execute
		// (the program may contain several kernels)
		//TODO
		
		// STUDENTS END

		// allocate global memory on GPU
		uint valuesSizeInBytes = m->nzNbr * sizeof(float);
		uint col_indSizeInBytes = m->nzNbr * sizeof(uint);
		uint row_ptrSizeInBytes = (m->h + 1) * sizeof(uint);
		uint vSizeInBytes = (v->h) * sizeof(float);
		uint mvSizeInBytes = (m->h) * sizeof(float);
		cl::Buffer gpuValues(context, CL_MEM_READ_ONLY, valuesSizeInBytes);
		cl::Buffer gpuCol_ind(context, CL_MEM_READ_ONLY, col_indSizeInBytes);
		cl::Buffer gpuRow_ptr(context, CL_MEM_READ_ONLY, row_ptrSizeInBytes);
		cl::Buffer gpuV(context, CL_MEM_READ_ONLY, vSizeInBytes);
		cl::Buffer gpuMV(context, CL_MEM_WRITE_ONLY, mvSizeInBytes); // result of matrix-vect multiplication

		// transfer data from CPU memory to GPU memory
		top(0); // start time measurement
		queue.enqueueWriteBuffer(gpuValues, CL_TRUE, 0, valuesSizeInBytes, m->data);
		queue.enqueueWriteBuffer(gpuCol_ind, CL_TRUE, 0, col_indSizeInBytes, m->col_ind);
		queue.enqueueWriteBuffer(gpuRow_ptr, CL_TRUE, 0, row_ptrSizeInBytes, m->row_ptr);
		queue.enqueueWriteBuffer(gpuV, CL_TRUE, 0, vSizeInBytes, v->data);
		queue.enqueueWriteBuffer(gpuMV, CL_TRUE, 0, mvSizeInBytes, mv->data);

		// set workgroup and grid size
		uint nbWarpsPerBlock = 1;
		size_t work_group_size = 32*nbWarpsPerBlock; // 1 warp = 32 threads;
		size_t global_work_size = ((int) ceilf(m->h*1.0/nbWarpsPerBlock)) * work_group_size;

		// STUDENTS BEGIN

		// set the arguments to our compute kernel
		//TODO
		//kernel.setArg(0, m->h);
		//kernel.setArg(6, sizeof(float)*work_group_size, NULL);
		
		// run kernel
		top(1);
		//TODO
		
		// Wait for the command queue to get serviced before reading back results
		queue.finish();
		gpuComputeTime = top(1); // pure computation duration

		// transfer data from GPU memory to CPU memory
		queue.enqueueReadBuffer(gpuMV, CL_TRUE, 0, mvSizeInBytes, mv->data);
		gpuRunTime = top(0); // computation and memory transfert duration
	}
	catch( cl::Error err )
	{
		std::cerr
			<< "ERROR: "
			<< err.what()
			<< "("
			<< err.err()
			<< ")"
			<< std::endl;

		if( err.what() == std::string("clBuildProgram") )
		{
			const char* error_type;
		
			if( err.err() == CL_INVALID_PROGRAM )
				error_type = "CL_INVALID_PROGRAM";
			else if( err.err() == CL_INVALID_VALUE )
				error_type = "CL_INVALID_VALUE";
			else if( err.err() == CL_INVALID_DEVICE )
				error_type = "CL_INVALID_DEVICE";
			else if( err.err() == CL_INVALID_BINARY )
				error_type = "CL_INVALID_BINARY";
			else if( err.err() == CL_INVALID_BUILD_OPTIONS )
				error_type = "CL_INVALID_BUILD_OPTIONS";
			else if( err.err() == CL_INVALID_OPERATION )
				error_type = "CL_INVALID_OPERATION";
			else if( err.err() ==  CL_COMPILER_NOT_AVAILABLE )
				error_type = " CL_COMPILER_NOT_AVAILABLE";
			else if( err.err() == CL_BUILD_PROGRAM_FAILURE )
				error_type = "CL_BUILD_PROGRAM_FAILURE";
			else if( err.err() == CL_INVALID_OPERATION )
				error_type = "CL_INVALID_OPERATION";
			else if( err.err() == CL_OUT_OF_HOST_MEMORY )
				error_type = "CL_OUT_OF_HOST_MEMORY";
		
			printf( "Program build error: %s.\n", error_type);
		}

		throw std::runtime_error("Aborting.");
	}
#endif

	// REMOVE FOR STUDENTS END

	// check result, display run time if result is correct
	bool displayRunTime = true;
	if( reference )
	{
		if(! checkResult(name, reference, mv))
			displayRunTime = false;
	}

	if(displayRunTime)
		printf("%s: M(%dx%d)xV computed in %f ms (%f ms of pure computation).\n", name, m->w, m->h, gpuRunTime, gpuComputeTime);

	return mv;
}

//---------------------------------------------------------
