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

uint rand_uint(u2* rvec);

inline float rand_float(u2* rvec);

__kernel void mainloop(__global pointcloud *pc, __global u2* randoms){
	size_t tid0 = get_global_id(0); // x
	size_t tid1 = get_global_id(1); // y

	const int wid = 1024;

	size_t pos = tid0 * wid;

	u2 rand1 = randoms[pos];
	u2 *rand = &rand1;
	
	affinetransform aff[4];
	for(int i = 0; i < 4; i++){
		aff[i].a = rand_float(rand);
		aff[i].b = rand_float(rand);
		aff[i].c = rand_float(rand);
		aff[i].d = rand_float(rand);
		aff[i].e = rand_float(rand);
		aff[i].f = rand_float(rand);
		aff[i].g = rand_float(rand);
		aff[i].h = rand_float(rand);
		aff[i].i = rand_float(rand);
		aff[i].j = rand_float(rand);
		aff[i].k = rand_float(rand);
		aff[i].l = rand_float(rand);
	}

	vec3 p = {1,1,1};
	col c = {0, 0, 0};
	for(int i = -20; i < wid; i++){
		affinetransform rand_aff = aff[rand_uint(rand) % 4];
		p = apply(rand_aff, p);
		
		float rsq = (p.x * p.x) + (p.y * p.y) + (p.z * p.z);
		float r = native_sqrt(rsq);

		vec3 p2;
		p2.x = p.x * sin(rsq) - p.y * cos(rsq);
		p2.y = p.x * cos(rsq) + p.y * sin(rsq);
		p2.z = p.z;

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

			pc[pos + i].r = (float)rand->x;
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


uint rand_uint(u2* rvec) {
    #define A 4294883355U
    uint x=rvec->x, c=rvec->y;
    uint res = x ^ c;
    uint hi = mul_hi(x,A);
    x = x*A + c;
    c = hi + (x<c);
    rvec->x = x;
    rvec->x = c;
    return res;
    #undef A
}

inline float rand_float(u2* rvec) {
    return (float)(rand_uint(rvec)) / (float)(0xFFFFFFFF);
}