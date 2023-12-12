#include "GPU_Ray.cuh"


__device__ __host__ void IntersectTri(Ray& ray, const Tri& tri)
{
	const glm::vec3 edge1 = tri.vertex1 - tri.vertex0;
	const glm::vec3 edge2 = tri.vertex2 - tri.vertex0;
	const glm::vec3 h = glm::cross(ray.D, edge2);

	const float a = dot(edge1, h);
	if (a > -0.0001f && a < 0.0001f)
		return; // Raza de lumina este paralela

	const float f = 1 / a;
	const glm::vec3 s = ray.O - tri.vertex0;
	const float u = f * glm::dot(s, h);
	if (u < 0 || u > 1) return;
	const glm::vec3 q = glm::cross(s, edge1);
	const float v = f * glm::dot(ray.D, q);
	if (v < 0 || u + v > 1) return;
	const float t = f * glm::dot(edge2, q);
	if (t > 0.0001f) ray.t = glm::min(ray.t, t);
}

__global__ void IntersectTri2(Ray& ray, Tri* tri)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	glm::vec3 edge1 = tri[i].vertex1 - tri[i].vertex0;
	glm::vec3 edge2 = tri[i].vertex2 - tri[i].vertex0;
	glm::vec3 h = glm::cross(ray.D, edge2);

	float a = dot(edge1, h);
	if (a > -0.0001f && a < 0.0001f)
		return; // Raza de lumina este paralela

	float f = 1 / a;
	glm::vec3 s = ray.O - tri[i].vertex0;
	float u = f * glm::dot(s, h);
	if (u < 0 || u > 1) return;
	glm::vec3 q = glm::cross(s, edge1);
	float v = f * glm::dot(ray.D, q);
	if (v < 0 || u + v > 1) return;
	float t = f * glm::dot(edge2, q);
	if (t > 0.0001f) ray.t = glm::min(ray.t, t);
}

__device__ void readPixel(char* pixels_gpu, int x, int y, unsigned char c, int width)
{
	pixels_gpu[x + y * width] = c;
}

__global__ void gpuRay_kernel(glm::vec3 camPos, glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, int width, int height, Ray* ray, Tri* tri, int N, char* pixels_gpu)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	readPixel(pixels_gpu, x, y, 0, width);

	glm::vec3 pixelPos = p0 + (p1 - p0) * (x / float(width)) + (p2 - p0) * (y / float(height));
	ray[x + y * width].O = camPos;
	ray[x + y * width].D = glm::normalize(pixelPos - ray[x + y * width].O);
	ray[x + y * width].t = 1e30f;

	for (int i = 0; i < N; i++)
		IntersectTri(ray[x + y * width], tri[i]);
	double a;
	a = sqrt(float(N) / 64.0);
	int blo = int(a);
	
	IntersectTri2<<<N/1024, 1024 >>>(ray[x + y * width], tri);
	//__syncthreads();
	

	if (ray[x + y * width].t < 1e30f) readPixel(pixels_gpu, x, y, 128, width);
}


void gpuRay(glm::vec3 camPos, glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, int width, int height, Ray* ray_gpu, Tri* tri_gpu, int N)
{
	int blocks;
	blocks = int(sqrt((width * height) / 64));
	std::cout << std::endl << std::endl << blocks << " blocuri";
	dim3 blocksPerGrid(blocks, blocks);
	dim3 threadsPerBlock(8, 8);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	gpuRay_kernel << <blocksPerGrid, threadsPerBlock >> > (camPos, p0, p1, p2, width, height, ray_gpu, tri_gpu, N, pixels_gpu);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float mytime;
	cudaEventElapsedTime(&mytime, start, stop);
	std::cout << std::endl << " GPU time: " << mytime;
	cudaMemcpy(pixels, pixels_gpu, bufferSize * sizeof(char), cudaMemcpyDeviceToHost);
}

void gpuRay_import(glm::vec3 camPos, glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, int width, int height, Ray* ray_gpu, std::vector<Tri> triangles, int N)
{
	int blocks;
	blocks = int(sqrt((width * height) / 64));
	std::cout << std::endl << std::endl << blocks << " blocuri";
	dim3 blocksPerGrid(blocks, blocks);
	dim3 threadsPerBlock(8, 8);

	Tri* copy_tri = (Tri*)malloc(sizeof(Tri) * triangles.size());
	Tri* triangles_gpu;

	for (unsigned int i = 0; i < triangles.size(); i++)
	{
		copy_tri[i] = triangles[i];
	}

	cudaMalloc((void**)&triangles_gpu, triangles.size() * sizeof(Tri));
	cudaMemcpy(triangles_gpu, copy_tri, triangles.size() * sizeof(Tri), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	gpuRay_kernel << <blocksPerGrid, threadsPerBlock >> > (camPos, p0, p1, p2, width, height, ray_gpu, triangles_gpu, triangles.size(), pixels_gpu);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float mytime;
	cudaEventElapsedTime(&mytime, start, stop);
	std::cout << std::endl <<" GPU time imported: " << mytime;
	cudaDeviceSynchronize();

	cudaMemcpy(pixels, pixels_gpu, bufferSize * sizeof(char), cudaMemcpyDeviceToHost);
}