#include "Math_Structs.cuh"

__device__ __host__ void IntersectTri(Ray& ray, const Tri& tri);

__device__ void readPixel(char* pixels_gpu, int x, int y, unsigned char c, int width);

__global__ void gpuRay_kernel(glm::vec3 camPos, glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, int width, int height, Ray* ray, Tri* tri, int N, char* pixels_gpu);

void gpuRay(glm::vec3 camPos, glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, int width, int height, Ray* ray_gpu, Tri* tri_gpu, int N);

void gpuRay_import(glm::vec3 camPos, glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, int width, int height, Ray* ray_gpu, std::vector<Tri> triangles, int N);