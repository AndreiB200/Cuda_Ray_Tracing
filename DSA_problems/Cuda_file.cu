#include "Cuda_head.cuh"

__device__ float r = 0.1f;
__device__ bool change = false;

__global__ void kernel(float *pos)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;

	if (x > 864)
		return;

	if (pos[x] > 1.0f && change == false)
	{
		r = -r;
		change = true;
	}
	if (pos[x] < -1.0f && change == true)
	{
		r = -r;
		change = false;
	}

	pos[x] += r;
	pos[x+1] += r;
	pos[x+2] += r;
}

cudaGraphicsResource* cuda_vbo;

void init_Cuda(unsigned int opengl_buffer)
{
	cudaGraphicsGLRegisterBuffer(&cuda_vbo, opengl_buffer, cudaGraphicsMapFlagsWriteDiscard);

	float* cuda_dev_pointer;
	cudaGraphicsMapResources(1, &cuda_vbo, 0);
	size_t mem_size;
	cudaGraphicsResourceGetMappedPointer((void**)&cuda_dev_pointer, &mem_size, cuda_vbo);
}

void launch_kernel()
{
	float* cuda_dev_pointer;
	cudaGraphicsMapResources(1, &cuda_vbo, 0);
	size_t mem_size;
	cudaGraphicsResourceGetMappedPointer((void**)&cuda_dev_pointer, &mem_size, cuda_vbo);

	int threads_per_block = 32;
	int blocks = (864 + threads_per_block - 1) / threads_per_block;

	kernel << <blocks,threads_per_block >> > (cuda_dev_pointer);
	cudaGraphicsUnmapResources(1, &cuda_vbo, 0);
}