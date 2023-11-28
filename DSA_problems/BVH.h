#include <iostream>
#include <vector>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <curand.h>

#include "Timer.h"

struct Ray
{
	glm::vec3 O, D; float t = 1e30f;
};

struct Tri
{
    glm::vec3 vertex0, vertex1, vertex2; glm::vec3 centroid;
};

class BVH
{
public:
    static const int N = 120;
	Tri tri[N];
    int width = 640, height = 640;

    int nrChannels, stride;
    char* pixels;

    //Initializare stricta pe CPU pt. salvarea imaginii in HDD --
    void getBuffer()
    {
        nrChannels = 1;
        stride = nrChannels * width;
        int bufferSize = stride * height;
        pixels = (char*)calloc(bufferSize, sizeof(char));
    }
    //Salvarea imaginii --
    void saveImg(char* filepath)
    {
        stbi_write_png(filepath, width, height, nrChannels, pixels, stride);
    }


    //Initializare Random facuta pe GPU impreuna cu "curand_init()"
    float RandomFloat()
    {
        float f = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        return f;
    }
    
    //Initializarea Triunghiurilor direct in placa video -> prin alocarea directa doar a marimii vectorului de triunghiuri
	void init()
	{
        for (int i = 0; i < N; i++)
        {
            glm::vec3 r0(RandomFloat(), RandomFloat(), RandomFloat());
            glm::vec3 r1(RandomFloat(), RandomFloat(), RandomFloat());
            glm::vec3 r2(RandomFloat(), RandomFloat(), RandomFloat());
            tri[i].vertex0 = r0 * 9.0f - glm::vec3(5);
            tri[i].vertex1 = tri[i].vertex0 + r1;
            tri[i].vertex2 = tri[i].vertex0 + r2;
        }

        shoot();
	}

    //Intersectarea logica a fiecarei Raze de lumina cu un triunghi la nivel de thread in GPU
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
    
    //Citirea Razelor de lumina facuta tot in GPU in memoria sa
    void readPixel(int x, int y, unsigned char c)
    {
        if (x < 0 || y < 0 || x >= width || y >= height) return;
            pixels[x + y * width] = c;
    }

    //Implementare paralela in GPU pentru fiecare Raza de lumina
    void shoot()
    {
        getBuffer();
        
        Timer time;
        time.startClock();

        glm::vec3 camPos(0, 0, -18);
        glm::vec3 p0(-1, 1, -15), p1(1, 1, -15), p2(-1, -1, -15);
        Ray ray;
        for (int y = 0; y < 640; y++) 
            for (int x = 0; x < 640; x++)
            {
                glm::vec3 pixelPos = p0 + (p1 - p0) * (x / 640.0f) + (p2 - p0) * (y / 640.0f);
                ray.O = camPos;
                ray.D = glm::normalize(pixelPos - ray.O);
                ray.t = 1e30f;
                for (int i = 0; i < N; i++) 
                    IntersectTri(ray, tri[i]);
                if (ray.t < 1e30f) readPixel(x, y, 255);
            }

        float timeFFF = time.stopClock(sec);
        std::cout << timeFFF;

        char saved[] = "./ray_trace.png";
        saveImg(saved);
    }
};

//Avand in vedere

//sudo apt pizda masi install