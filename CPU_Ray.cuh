#include "Math_Structs.cuh"


class CPU_Ray
{
public: //Variables
    int width = 1024, height = 1024;

    static const int N = 12;
    Tri* tri = (Tri*)malloc(sizeof(Tri) * N);
    Ray* ray = (Ray*)malloc(sizeof(Ray) * width * height);

    int nrChannels, stride;


public: //Functions
    void init();


    void getBuffer();

    void saveImg(char* filepath, char* pixels);

    float RandomFloat();

    void IntersectTri(Ray& ray, const Tri& tri);


    void readPixel(int x, int y, unsigned char c);

    void intersectionTri(Ray& ray);


    void multiParRay(glm::vec3 camPos, glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, int y);

    void shoot();
};