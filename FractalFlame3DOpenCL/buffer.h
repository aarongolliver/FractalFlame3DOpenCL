#ifndef __BUFFER
#define __BUFFER

#include "GenVector.h"

class ColBuffer
{
public:
	ColBuffer() : w(0), h(0) {
		this->data = NULL;
	}

	ColBuffer(unsigned int _w, unsigned int _h) : w(_w), h(_h)
	{
		alloc();
	}

	Color at(unsigned int x, unsigned int y) const
	{
		return data[x + y*this->w];
	}

	Color & at(unsigned int x, unsigned int y)
	{
		return data[x + y*this->w];
	}

private:
	const unsigned int w;
	const unsigned int h;
	Color * data;

	void alloc()
	{
		size_t size = this->w * this->h;
		if (size == 0){
			this->data = NULL;
			return;
		}

		data = (Color*) calloc(this->w * this->h, sizeof(Color));
	}
};

typedef struct {
	float r, g, b;
	unsigned int a;
} histogram;

class HistoBuffer
{
public:
	HistoBuffer() : w(0), h(0) {
		this->data = NULL;
	}

	HistoBuffer(unsigned int _w, unsigned int _h) : w(_w), h(_h)
	{
		alloc();
	}

	histogram at(unsigned int x, unsigned int y) const
	{
		return data[x + y*this->w];
	}

	histogram & at(unsigned int x, unsigned int y)
	{
		return data[x + y*this->w];
	}

private:
	const unsigned int w;
	const unsigned int h;
	histogram * data;

	void alloc()
	{
		size_t size = this->w * this->h;
		if (size == 0){
			this->data = NULL;
			return;
		}

		data = (histogram*)calloc(this->w * this->h, sizeof(histogram));
	}
};
#endif