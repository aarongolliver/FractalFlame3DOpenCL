#define n_kernels (1024 * 8)
#define points_kernel (1024 * 6)

#define pc_size (n_kernels * points_kernel)

#define rand_buf_size (n_kernels * points_kernel)

typedef struct {
	float x, y, z, r, g, b;
} pointcloud;

typedef struct {
	float x, y, z;
} vec3;

typedef struct {
	float r, g, b;
} col;

typedef struct {
	uint x;
	uint y;
} u2;

typedef struct {
	float a, b, c, d, e, f, g, h, i, j, k, l;
	float red, gre, blu;
} affinetransform;

vec3 apply(const affinetransform aff, const vec3 p);



uint rand_uint(int *rand_offset, __global uint *randoms, uint rand_xor){
	(*rand_offset)++;
	*rand_offset = *rand_offset % rand_buf_size;

	return (*(randoms + *rand_offset)) ^ rand_xor;
}

#define rand_float(offs, rand, rand_xor) ((float)(rand_uint(offs, rand, rand_xor)) / (float)(0xFFFFFFFF))

__kernel void mainloop(__global pointcloud *pc, __global uint *randoms, __global affinetransform *affs){
	size_t tid0 = get_global_id(0); // x

	size_t pos = tid0 * points_kernel;
	int rand_offset = pos;

	__global uint *rand = randoms;
	uint rand_xor = *(rand + rand_offset++);

	vec3 p = {1,1,1};
	col c = {0, 0, 0};
	for(int i = -200; i < points_kernel; i++){
		affinetransform rand_aff = affs[rand_uint(&rand_offset, rand, rand_xor) % 5];
		p = apply(rand_aff, p);
		
		float rsq = (p.x * p.x) + (p.y * p.y) + (p.z * p.z);
		float r = native_sqrt(rsq);

		vec3 p2;
		p2.x = p.x * sin(rsq) - p.y * cos(rsq);
		p2.y = p.x * cos(rsq) + p.y * sin(rsq);
		p2.z = p.z;


		p2.x = sin(p.x);
		p2.y = sin(p.y);
		p2.z = sin(p.z);

		p = p2;
		c.r += rand_aff.red;
		c.g += rand_aff.gre;
		c.b += rand_aff.blu;
		
		c.r /= 2.0f;
		c.g /= 2.0f;
		c.b /= 2.0f;

		if(i >= 0){
			pc[pos + i].x = p.x;
			pc[pos + i].y = p.y;
			pc[pos + i].z = p.z;

			pc[pos + i].r = c.r;
			pc[pos + i].g = c.g;
			pc[pos + i].b = c.b;	
		}
	}

	//pc[pos].x = p.x;
	//pc[pos].y = p.y;
	//pc[pos].z = p.z;
	//pc[pos].r = tid0;
	//pc[pos].g = p.y;
	//pc[pos].b = p.z;
}

vec3 apply(const affinetransform aff, const vec3 p) {
	vec3 ret;
	ret.x = (p.x * aff.a) + (p.y * aff.b) + (p.z * aff.c) + (aff.d);
	ret.y = (p.x * aff.e) + (p.y * aff.f) + (p.z * aff.g) + (aff.h);
	ret.z = (p.x * aff.i) + (p.y * aff.j) + (p.z * aff.k) + (aff.l);
	return ret;
}