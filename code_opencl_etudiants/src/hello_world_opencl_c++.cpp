
#define __CL_ENABLE_EXCEPTIONS

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
//#include <CL/cl.hpp>
#include "cl.hpp"
#endif

#include <cstdio>
#include <cstdlib>
#include <iostream>

#define DATA_SIZE (1024)
 
//-------------------------------------------------------------------

// Simple compute kernel which computes the square of an input array 
std::string kernel_source =
	"__kernel void square(\n"
	"   __global float* input,\n" 
	"   __global float* output,\n" 
	"   const unsigned int count)\n" 
	"{\n" 
	"   int i = get_global_id(0);\n" 
	"   if(i < count)\n" 
	"       output[i] = input[i] * input[i];\n" 
	"}\n"; 

//-------------------------------------------------------------------

int main(void)
{
    // prepare data with random float values
    float data[DATA_SIZE];              // original data set given to device
    float results[DATA_SIZE];           // results returned from device
    unsigned int count = DATA_SIZE;
    for( uint i = 0; i < count; i++)
        data[i] = rand() / (float)RAND_MAX;

	try 
	{
		// init OpenCL

		// retreive list of available platforms and select the first one
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		if( platforms.size() == 0 ) 
		{
			std::cout << "No OpenCL platform found. Check installation!\n";
			return -1;
		}
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
		sources.push_back(std::make_pair(kernel_source.c_str(), kernel_source.length()));
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
		cl::Kernel kernel(program, "square");

		// allocate global memory on GPU
		cl::Buffer gpu_input(context, CL_MEM_READ_ONLY, sizeof(float)*count);
		cl::Buffer gpu_output(context, CL_MEM_WRITE_ONLY, sizeof(float)*count);

		// Write our data set into the input array in device memory 
		queue.enqueueWriteBuffer(gpu_input, CL_TRUE, 0, sizeof(float)*count, data);

		// Set the arguments to our compute kernel
		kernel.setArg(0, gpu_input);
		kernel.setArg(1, gpu_output);
		kernel.setArg(2, count);

		// get the maximum work group size for executing the kernel on the device
		// real max workgroup size depends on kernel complexity
		size_t work_group_size;
		kernel.getWorkGroupInfo(device, CL_KERNEL_WORK_GROUP_SIZE, &work_group_size);

		// Execute the kernel over the entire range of our 1d input data set
		// using the maximum number of work group items for this device
		size_t global_work_size = count;
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_work_size), cl::NDRange(work_group_size));

		// Wait for the command queue to get serviced before reading back results
		queue.finish();

		// Read back the results from the device to verify the output
		queue.enqueueReadBuffer(gpu_output, CL_TRUE, 0, sizeof(float)*count, results);
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

		return EXIT_FAILURE;
	}

	// Validate our results
	unsigned int correct = 0;
	for( uint i = 0; i < count; i++)
	{
		if( results[i] == data[i]*data[i] )
			correct++;
	}

	// Print a brief summary detailing the results
	printf("Computed '%d/%d' correct values!\n", correct, count);

	return EXIT_SUCCESS;
}
