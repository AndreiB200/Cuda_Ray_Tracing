#include <glad/glad.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>

void init_Cuda(unsigned int opengl_buffer);
void launch_kernel();