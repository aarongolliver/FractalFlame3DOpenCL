#ifndef MAIN_H
#define MAIN_H

#include <iostream>
#include <cstdio>
#include <cstdlib>

inline void
checkErr(cl_int err, const char * name)
{
	if (err != CL_SUCCESS) {
		std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
		getchar();
		exit(EXIT_FAILURE);
	}
}

typedef struct {
	float x, y, z, r, g, b;
} pointcloud;

typedef struct {
	unsigned int x;
	unsigned int y;
} u2;

#endif