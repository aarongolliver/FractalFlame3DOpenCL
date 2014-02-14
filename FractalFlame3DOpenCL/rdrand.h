#ifndef __RDRAND_H__
#define __RDRAND_H__ __RDRAND_H__

#include "datatypes.h"
#include <stdio.h>
#include <stdlib.h>

#define RDRAND_SUCCESS 1

__forceinline void rdrand_u16(u16 *p){
	while (_rdrand16_step(p) != RDRAND_SUCCESS);
}

//__forceinline void rdrand_u32(u32 *p){
//	while (_rdrand32_step(p) != RDRAND_SUCCESS);
//}

__forceinline void rdrand_u32(u32 *p){
	float t = ((float)rand()) / RAND_MAX;
	*p = (u32)(t * 0xFFFFFFFFu);
	*p = (*p) ^ rand();
}


// floating point random functions return (-1, 1)
__forceinline void rdrand_f32(f32 *p){
	*p = ((float)rand()) / RAND_MAX;
	if (rand() % 2)
		*p = (*p) * -1;
}

#endif