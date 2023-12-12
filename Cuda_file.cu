//#include "Cuda_head.cuh"
//
//__device__ float r = 0.1f;
//__device__ bool change = false;
//
//__global__ void kernel(float *pos)
//{
//	int x = blockDim.x * blockIdx.x + threadIdx.x;
//
//	if (x > 864)
//		return;
//
//	if (pos[x] > 1.0f && change == false)
//	{
//		r = -r;
//		change = true;
//	}
//	if (pos[x] < -1.0f && change == true)
//	{
//		r = -r;
//		change = false;
//	}
//
//	pos[x] += r;
//	pos[x+1] += r;
//	pos[x+2] += r;
//}
//
//__device__ __host__ void IntersectTri(Ray& ray, const Tri& tri)
//{
//	const glm::vec3 edge1 = tri.vertex1 - tri.vertex0;
//	const glm::vec3 edge2 = tri.vertex2 - tri.vertex0;
//	const glm::vec3 h = glm::cross(ray.D, edge2);
//
//	const float a = dot(edge1, h);
//	if (a > -0.0001f && a < 0.0001f)
//		return; // Raza de lumina este paralela
//
//	const float f = 1 / a;
//	const glm::vec3 s = ray.O - tri.vertex0;
//	const float u = f * glm::dot(s, h);
//	if (u < 0 || u > 1) return;
//	const glm::vec3 q = glm::cross(s, edge1);
//	const float v = f * glm::dot(ray.D, q);
//	if (v < 0 || u + v > 1) return;
//	const float t = f * glm::dot(edge2, q);
//	if (t > 0.0001f) ray.t = glm::min(ray.t, t);
//}
//
//__device__ void readPixel(char* pixels_gpu, int x, int y, unsigned char c)
//{
//	//if (x < 0 || y < 0 || x >= width || y >= height) return;
//	pixels_gpu[x + y * 640] = c;
//}
//
//__global__ void gpuRay_kernel(glm::vec3 camPos, glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, int width, int height, Ray* ray, Tri* tri, int N, char* pixels_gpu)
//{
//	int x = blockDim.x * blockIdx.x + threadIdx.x;
//	int y = blockDim.y * blockIdx.y + threadIdx.y;
//
//	glm::vec3 pixelPos = p0 + (p1 - p0) * (x / float(width)) + (p2 - p0) * (y / float(height));
//	ray[x + y * width].O = camPos;
//	ray[x + y * width].D = glm::normalize(pixelPos - ray[x + y * width].O);
//	ray[x + y * width].t = 1e30f;
//
//	for (int i = 0; i < N; i++)
//		IntersectTri(ray[x + y * width], tri[i]);
//
//	if (ray[x + y * width].t < 1e30f) readPixel(pixels_gpu, x, y, 255);
//}
//
//
//void gpuRay(glm::vec3 camPos, glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, int width, int height, Ray* ray_gpu, Tri* tri_gpu, int N)
//{
//	dim3 blocksPerGrid(80, 80);
//	dim3 threadsPerBlock(8, 8);
//	gpuRay_kernel << <blocksPerGrid, threadsPerBlock >> > (camPos, p0, p1, p2, width, height, ray_gpu, tri_gpu, N, pixels_gpu);
//}
//
//
//cudaGraphicsResource* cuda_vbo;
//void init_Cuda(unsigned int opengl_buffer)
//{
//	cudaGraphicsGLRegisterBuffer(&cuda_vbo, opengl_buffer, cudaGraphicsMapFlagsWriteDiscard);
//
//	float* cuda_dev_pointer;
//	cudaGraphicsMapResources(1, &cuda_vbo, 0);
//	size_t mem_size;
//	cudaGraphicsResourceGetMappedPointer((void**)&cuda_dev_pointer, &mem_size, cuda_vbo);
//}
//void launch_kernel()
//{
//	float* cuda_dev_pointer;
//	cudaGraphicsMapResources(1, &cuda_vbo, 0);
//	size_t mem_size;
//	cudaGraphicsResourceGetMappedPointer((void**)&cuda_dev_pointer, &mem_size, cuda_vbo);
//
//	int threads_per_block = 32;
//	int blocks = (864 + threads_per_block - 1) / threads_per_block;
//
//	kernel << <blocks,threads_per_block >> > (cuda_dev_pointer);
//	cudaGraphicsUnmapResources(1, &cuda_vbo, 0);
//}
