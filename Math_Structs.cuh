#ifndef MATH_STRUCTS_H
#define MATH_STRUCTS_H

#include <glad/glad.h>

#include <iostream>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Timer.h"
#include "Thread_Pool.h"

struct Ray
{
	glm::vec3 O, D; float t = 1e30f;
};

struct Tri
{
	glm::vec3 vertex0, vertex1, vertex2; glm::vec3 centroid;
};

extern char* pixels_gpu;
extern char* pixels_from_gpu;
extern char* pixels;
extern int bufferSize;

extern Tri* tri_gpu;
extern Ray* ray_gpu;

extern std::vector<Tri> triangles;

#endif