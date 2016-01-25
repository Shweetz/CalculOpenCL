//
// File:       hello.c
//
// Abstract:   A simple "Hello World" compute example showing basic usage of OpenCL which
//             calculates the mathematical square (X[i] = pow(X[i],2)) for a buffer of
//             floating point values.
//             
//
// Version:    <1.0>
//
// Disclaimer: IMPORTANT:  This Apple software is supplied to you by Apple Inc. ("Apple")
//             in consideration of your agreement to the following terms, and your use,
//             installation, modification or redistribution of this Apple software
//             constitutes acceptance of these terms.  If you do not agree with these
//             terms, please do not use, install, modify or redistribute this Apple
//             software.
//
//             In consideration of your agreement to abide by the following terms, and
//             subject to these terms, Apple grants you a personal, non - exclusive
//             license, under Apple's copyrights in this original Apple software ( the
//             "Apple Software" ), to use, reproduce, modify and redistribute the Apple
//             Software, with or without modifications, in source and / or binary forms;
//             provided that if you redistribute the Apple Software in its entirety and
//             without modifications, you must retain this notice and the following text
//             and disclaimers in all such redistributions of the Apple Software. Neither
//             the name, trademarks, service marks or logos of Apple Inc. may be used to
//             endorse or promote products derived from the Apple Software without specific
//             prior written permission from Apple.  Except as expressly stated in this
//             notice, no other rights or licenses, express or implied, are granted by
//             Apple herein, including but not limited to any patent rights that may be
//             infringed by your derivative works or by other works in which the Apple
//             Software may be incorporated.
//
//             The Apple Software is provided by Apple on an "AS IS" basis.  APPLE MAKES NO
//             WARRANTIES, EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION THE IMPLIED
//             WARRANTIES OF NON - INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A
//             PARTICULAR PURPOSE, REGARDING THE APPLE SOFTWARE OR ITS USE AND OPERATION
//             ALONE OR IN COMBINATION WITH YOUR PRODUCTS.
//
//             IN NO EVENT SHALL APPLE BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL OR
//             CONSEQUENTIAL DAMAGES ( INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//             SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//             INTERRUPTION ) ARISING IN ANY WAY OUT OF THE USE, REPRODUCTION, MODIFICATION
//             AND / OR DISTRIBUTION OF THE APPLE SOFTWARE, HOWEVER CAUSED AND WHETHER
//             UNDER THEORY OF CONTRACT, TORT ( INCLUDING NEGLIGENCE ), STRICT LIABILITY OR
//             OTHERWISE, EVEN IF APPLE HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright ( C ) 2008 Apple Inc. All Rights Reserved.
//

////////////////////////////////////////////////////////////////////////////////

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/opencl.h>
 
 
#define DATA_SIZE (1024)
 
//-------------------------------------------------------------------

// Simple compute kernel which computes the square of an input array 
const char *kernel_source =
	"__kernel void square(\n"
	"   __global float* input,\n" 
	"   __global float* output,\n" 
	"   const unsigned int count)\n" 
	"{\n" 
	"   bla int i = get_global_id(0);\n" 
	"   if(i < count)\n" 
	"       output[i] = input[i] * input[i];\n" 
	"}\n"; 

//-------------------------------------------------------------------

int main(int argc, char** argv)
{
    // prepare data with random float values
    float data[DATA_SIZE];              // original data set given to device
    float results[DATA_SIZE];           // results returned from device
    unsigned int count = DATA_SIZE;
    for( uint i = 0; i < count; i++)
        data[i] = rand() / (float)RAND_MAX;

	// to store temporary log messages or other informations
	char buffer[4096];

	// init OpenCL

    cl_int err; // error code returned from api calls
      
	// retreive list of available platforms
	cl_platform_id platform_id[10];
	cl_uint num_of_platforms = 0;
	err = clGetPlatformIDs( sizeof(platform_id), platform_id, &num_of_platforms);
	if( err != CL_SUCCESS )
	{
		printf("Unable to get platform_ids\n");
		return -1;
	}

	// display first platform name
	clGetPlatformInfo( platform_id[0], CL_PLATFORM_NAME, sizeof(buffer), (void*) buffer, NULL);
	printf("Using platform: %s\n", buffer);

	// get the first GPU device
    cl_device_id device_id;
	cl_uint num_of_devices = 0;
	err = clGetDeviceIDs( platform_id[0], CL_DEVICE_TYPE_GPU, 1, &device_id, &num_of_devices);
	if( err != CL_SUCCESS )
	{
		printf("Unable to get GPU device_id\n");
		return -1;
	}

	// display informations on device
	printf("Using device :\n");
	clGetDeviceInfo( device_id, CL_DEVICE_NAME, sizeof(buffer), (void*) buffer, NULL);
	printf( "  CL_DEVICE_NAME    = %s\n", buffer);
	clGetDeviceInfo( device_id, CL_DEVICE_VENDOR, sizeof(buffer), (void*) buffer, NULL);
	printf( "  CL_DEVICE_VENDOR  = %s\n", buffer);
	clGetDeviceInfo( device_id, CL_DEVICE_VERSION, sizeof(buffer), (void*) buffer, NULL);
	printf( "  CL_DEVICE_VERSION = %s\n", buffer);
	clGetDeviceInfo( device_id, CL_DRIVER_VERSION, sizeof(buffer), (void*) buffer, NULL);
	printf( "  CL_DRIVER_VERSION = %s\n", buffer);

	// context properties list - must be terminated with 0
	cl_context_properties properties[3];
	properties[0]= CL_CONTEXT_PLATFORM;
	properties[1]= (cl_context_properties) platform_id[0];
	properties[2]= 0;

	// create a context with the GPU device
    cl_context context = clCreateContext( properties, 1, &device_id, NULL, NULL, &err);
	if( err != CL_SUCCESS )
	{
		printf("Failed to create context.\n");
		return -1;
	}

	// create command queue using the context and device
    cl_command_queue command_queue = clCreateCommandQueue( context, device_id, 0, &err);
	if( err != CL_SUCCESS )
	{
		printf("Failed to create command queue.\n");
		return -1;
	}
	
	// init ok
	printf("Compute device successfully initialized.\n");

	// compile kernel

	// create a program from the kernel source code
    cl_program program = clCreateProgramWithSource( context, 1, &kernel_source, NULL, NULL);
	if( program == NULL )
	{
		printf("Failed to create program with source.\n");
		return -1;
	}

	// compile program
	const char *options = "";
	err = clBuildProgram( program, 1, &device_id, options, NULL, NULL);

	// get and display build log
	size_t log_size = 0;
	clGetProgramBuildInfo( program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &log_size);
	if( log_size > 2 )
	{
		printf("Program build log:\n"); 
		printf("%s\n",buffer);
	}
	if( err != CL_SUCCESS)
	{
		const char* error_type;
	
		if( err == CL_INVALID_PROGRAM )
			error_type = "CL_INVALID_PROGRAM";
		else if( err == CL_INVALID_VALUE )
			error_type = "CL_INVALID_VALUE";
		else if( err == CL_INVALID_DEVICE )
			error_type = "CL_INVALID_DEVICE";
		else if( err == CL_INVALID_BINARY )
			error_type = "CL_INVALID_BINARY";
		else if( err == CL_INVALID_BUILD_OPTIONS )
			error_type = "CL_INVALID_BUILD_OPTIONS";
		else if( err == CL_INVALID_OPERATION )
			error_type = "CL_INVALID_OPERATION";
		else if( err ==  CL_COMPILER_NOT_AVAILABLE )
			error_type = " CL_COMPILER_NOT_AVAILABLE";
		else if( err == CL_BUILD_PROGRAM_FAILURE )
			error_type = "CL_BUILD_PROGRAM_FAILURE";
		else if( err == CL_INVALID_OPERATION )
			error_type = "CL_INVALID_OPERATION";
		else if( err == CL_OUT_OF_HOST_MEMORY )
			error_type = "CL_OUT_OF_HOST_MEMORY";
	
		printf( "Program build error: %s.\n", error_type);
		return -1;
	}
	else
	{
		printf("Program successfully built.\n");
	}

	// specify which kernel to execute
	// (the program may contain several kernels)
	cl_kernel kernel = clCreateKernel( program, "square", NULL);
	if( kernel == NULL )
	{
		printf("Error: failed to get kernel.\n");
		return -1;
	}

	// allocate global memory on GPU
	cl_mem gpu_input = clCreateBuffer( context, CL_MEM_READ_ONLY, sizeof(float)*count, NULL, NULL);
	cl_mem gpu_output = clCreateBuffer( context, CL_MEM_WRITE_ONLY, sizeof(float)*count, NULL, NULL);
	if( gpu_input == NULL  ||  gpu_output == NULL )
	{
		printf("Failed to allocate memory on GPU.\n");
		return -1;
	}

    // Write our data set into the input array in device memory 
    err = clEnqueueWriteBuffer( command_queue, gpu_input, CL_TRUE, 0, sizeof(float)*count, data, 0, NULL, NULL);
    if( err != CL_SUCCESS )
    {
        printf("Error: Failed to write data to GPU memory!\n");
        exit(1);
    }
 
    // Set the arguments to our compute kernel
    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &gpu_input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &gpu_output);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &count);
    if( err != CL_SUCCESS )
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }
 
    // get the maximum work group size for executing the kernel on the device
	// real max workgroup size depends on kernel complexity
    size_t work_group_size;
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(work_group_size), &work_group_size, NULL);
    if( err != CL_SUCCESS )
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }
 
    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    size_t global_work_size = count;
    err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &work_group_size, 0, NULL, NULL);
    if( err )
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }
 
    // Wait for the command queue to get serviced before reading back results
    clFinish(command_queue);
 
    // Read back the results from the device to verify the output
    err = clEnqueueReadBuffer( command_queue, gpu_output, CL_TRUE, 0, sizeof(float)*count, results, 0, NULL, NULL);  
    if (err != CL_SUCCESS)
    {
        printf( "Error: Failed to read output array in GPU memory! %d\n", err);
        exit(1);
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
    
    // Shutdown and cleanup
    clReleaseMemObject(gpu_input);
    clReleaseMemObject(gpu_output);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
 
    return 0;
}
