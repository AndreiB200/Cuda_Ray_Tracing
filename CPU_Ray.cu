#include "CPU_Ray.cuh"
#include "stb_image_write.h"

#include "GPU_Ray.cuh"

char* pixels_gpu;
char* pixels;

Tri* tri_gpu;
Ray* ray_gpu;

int bufferSize;

Thread_Pool threads;

void CPU_Ray::getBuffer()
{
    nrChannels = 1;
    stride = nrChannels * width;
    bufferSize = stride * height;
    pixels = (char*)calloc(bufferSize, sizeof(char));

    cudaMalloc((void**)&pixels_gpu, bufferSize * sizeof(char));
    cudaMemcpy(pixels_gpu, pixels, bufferSize * sizeof(char), cudaMemcpyHostToDevice);
}

void CPU_Ray::saveImg(char* filepath, char* pixels)
{
    stbi_write_png(filepath, width, height, nrChannels, pixels, stride);
}

float CPU_Ray::RandomFloat()
{
    float f = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    return f;
}

void CPU_Ray::init()
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

    cudaMalloc((void**)&tri_gpu, N * sizeof(Tri));
    cudaMemcpy(tri_gpu, tri, N * sizeof(Tri), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&ray_gpu, width * height * sizeof(Ray));
    cudaMemcpy(ray_gpu, ray, width * height * sizeof(Ray), cudaMemcpyHostToDevice);

    shoot();
}

void CPU_Ray::IntersectTri(Ray& ray, const Tri& tri)
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

void CPU_Ray::readPixel(int x, int y, unsigned char c)
{
    //if (x < 0 || y < 0 || x >= width || y >= height) return;
    pixels[x + y * width] = c;
}

void CPU_Ray::intersectionTri(Ray& ray)
{
    for (int i = 0; i < N; i++)
        IntersectTri(ray, tri[i]);
}

void CPU_Ray::multiParRay(glm::vec3 camPos, glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, int y)
{
    for (int x = 0; x < width; x++)
    {
        glm::vec3 pixelPos = p0 + (p1 - p0) * (x / float(width)) + (p2 - p0) * (y / float(height));
        ray[x + y * width].O = camPos;
        ray[x + y * width].D = glm::normalize(pixelPos - ray[x + y * width].O);
        ray[x + y * width].t = 1e30f;

        for (int i = 0; i < N; i++)
            IntersectTri(ray[x + y * width], tri[i]);

        if (ray[x + y * width].t < 1e30f) readPixel(x, y, 255);
    }
}

void CPU_Ray::shoot()
{
    getBuffer();

    Timer time;
    time.startClock();

    glm::vec3 camPos(0, 0, -18);
    glm::vec3 p0(-1, 1, -15), p1(1, 1, -15), p2(-1, -1, -15);

    for (int y = 0; y < height; y++)
        threads.addJob(std::bind(&CPU_Ray::multiParRay, this, camPos, p0, p1, p2, y));

    threads.wait();
    threads.stop();
    float timeFFF = time.stopClock(sec);
    std::cout << std::endl << " CPU time: " << timeFFF << std::endl;
    char saved[] = "./ray_trace.png";
    char saved2[] = "./rat_trace_gpu.png";
    char saved3[] = "./rat_trace_gpu_import.png";

    saveImg(saved, pixels);

    gpuRay(camPos, p0, p1, p2, width, height, ray_gpu, tri_gpu, N);

    saveImg(saved2, pixels);

    gpuRay_import(camPos, p0, p1, p2, width, height, ray_gpu, triangles, N);

    saveImg(saved3, pixels);
}