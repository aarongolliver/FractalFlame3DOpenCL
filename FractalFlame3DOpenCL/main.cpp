#include <utility>
#include <CL/cl.hpp>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <iterator>

#include "main.h"

using namespace std;



int main(int argc, char **argv){
	cl_int err;
	
	// get all the availiable platforms
	vector< cl::Platform > platformList;
	cl::Platform::get(&platformList);

	checkErr(platformList.size() != 0 ? CL_SUCCESS : -1, "cl::Platform::get");

	cout << "Platform number is: " << platformList.size() << endl;
	string platformVendor;
	platformList[0].getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platformVendor);
	cout << "Platform Vendor is " << platformVendor << endl;

	// create opencl context
	cl_context_properties cprops[3] = {
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)(platformList[0])(),
		0
	};

	cl::Context context(
		CL_DEVICE_TYPE_GPU,
		cprops,
		NULL,
		NULL,
		&err);

	checkErr(err, "Context::Context()");

	pointcloud *pcFront = new pointcloud[1024 * 1024];
	memset(pcFront, 0, sizeof(pointcloud)* 1024 * 1024);

	// create the front and back buffers
	cl::Buffer frontBuffer(
		context,
		CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
		sizeof(pointcloud)* 1024 * 1024,
		NULL,
		&err);

	checkErr(err, "Buffer::Buffer() (front buffer) ");
	// create the rand buffer
	u2 *rand_buf = new u2[1024 * 1024];
	for (int i = 0; i < 1024 * 1024; i++){
		rand_buf[i].x = rand();
		rand_buf[i].y = rand();
	}

	cl::Buffer cl_rand_buf(
		context,
		CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
		sizeof(u2)* 1024 * 1024,
		NULL,
		&err);

	// set up devices more?
	vector< cl::Device> devices;
	devices = context.getInfo<CL_CONTEXT_DEVICES>();
	checkErr(devices.size() > 0 ? CL_SUCCESS : -1, "devices.size() > 0");

	// load and compile the OpenCL
	ifstream kernel_file("FractalEngine.cl");
	checkErr(kernel_file.is_open() ? CL_SUCCESS : -1, "FractalEngine.cl failed to open");
	string prog(istreambuf_iterator<char>(kernel_file), (std::istreambuf_iterator<char>()));

	cl::Program::Sources source(
		1,
		make_pair(prog.c_str(), prog.length() + 1));
	cl::Program program(context, source);
	err = program.build(devices, "");
	cl::STRING_CLASS buildLog;

	checkErr(err, "Program::build()");

	// create the kernel, give it frontBuffer as a parameter
	cl::Kernel kernel(program, "mainloop", &err);
	checkErr(err, "Kernel::Kernel()");
	err = kernel.setArg(0, frontBuffer);
	err = kernel.setArg(1, cl_rand_buf);
	checkErr(err, "Kernel::setArg(frontBuffer)");

	// create the command queues
	cl::CommandQueue queue(context, devices[0], 0, &err);
	checkErr(err, "CommandQueue::CommandQueue()");

	cl::Event frontBufferEvent;

	queue.enqueueWriteBuffer(
		cl_rand_buf, CL_TRUE, 0, sizeof(u2)* 1024 * 1024, rand_buf);
	
	err = queue.enqueueNDRangeKernel(
		kernel,
		cl::NullRange,
		cl::NDRange(1024),
		cl::NDRange(16, 16),
		NULL, // this would be a vector to events that must be completed before this starts! queue up the RNG here :)
		&frontBufferEvent);
	checkErr(err, "CommandQueue::enqueueNDRangeKernel()");

	frontBufferEvent.wait();
	err = queue.enqueueReadBuffer(
		frontBuffer,
		CL_TRUE, // blocking
		0,
		sizeof(pointcloud)* 1024 * 1024,
		pcFront);
	checkErr(err, "CommandQueue::enqueueReadBuffer()");
	
	for (int i = 0; i < 1024; i++)
		cout << pcFront[i].r << endl;

	getchar();

	exit(EXIT_SUCCESS);
}