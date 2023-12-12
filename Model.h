#ifndef MODEL_H
#define MODEL_H

#include <glad/glad.h> 
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>


#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "Mesh.h"
#include "Shader.h"

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <functional>

#include "Thread_Pool.h"
#include "Math_Structs.cuh"

using namespace std;

std::vector<Tri> triangles;

enum axis
{
    X, Y, Z,
};

class Model
{
public:
    bool model_loaded = false;
    
    vector<Mesh>    meshes;
    string          directory;
    bool            gammaCorrection;

    glm::mat4 model = glm::mat4(1.0f);
    glm::vec3 position = glm::vec3(0.0f);
    glm::vec3 size = glm::vec3(1.0f);
    float axisX, axisY, axisZ, degrees;
    glm::vec3 rotation = glm::vec3(0.0f);

    unsigned int totalPoly = 0;
    vector<vector<Vertex>> allVertices;
    vector<vector<unsigned int>> allIndices;

    string path;

    void move(float x, float y, float z)
    {
        position.x = x; position.y = y; position.z = z;
        model = glm::translate(model, position);
    }
    void move(glm::vec3 v_position)
    {
        position = v_position;
        model = glm::translate(model, position);
    }
    void move()
    {
        model = glm::translate(model, position);
    }
    
    void scale(float x, float y, float z)
    {
        size.x = x; size.y = y; size.z = z;
        model = glm::scale(model, size);
    }
    void scale(float x)
    {
        size = glm::vec3(x);
        model = glm::scale(model, size);
    }
    
    void rotate(float m_degrees, axis axisR)
    {
        degrees = m_degrees;
        if (axisR == X)
            rotation = glm::vec3(1.0f, 0.0f, 0.0f);
        if (axisR == Y)
            rotation = glm::vec3(0.0f, 1.0f, 0.0f);
        if (axisR == Z)
            rotation = glm::vec3(0.0f, 0.0f, 1.0f);

        model = glm::rotate(model, glm::radians(m_degrees), rotation);
    }

    
    
    void acces()
    {
        //std::cout << "Acest model are " << totalPoly << "poligoane cu ID: "<< meshes[0].VAO << std::endl;
    }

    Model(string const& paath, bool gamma = false) : gammaCorrection(gamma)
    {
        path = paath;
        std::function<void()> f = std::bind(&Model::loadModel, this);
        
        
        f();

        std::cout << "Thread initializat " << std::endl;
    }
    //Model() {}

    void applyData()
    {        
        loadMeshes();
    }



private:
    void loadMeshes()
    {
        for (unsigned int i = 0; i < allVertices.size(); i++)
        {
            meshes.push_back(Mesh(allVertices[i], allIndices[i]));
            totalPoly = totalPoly + allVertices[i].size();
        }
        std::cout << "Acest model are " << totalPoly << " poligoane cu ID: " << meshes[0].VAO << std::endl;
        allVertices.clear();
        allVertices.shrink_to_fit();
        allIndices.clear();
        allVertices.shrink_to_fit();
        std::cout << "Modelul este incarcat in GPU !" << std::endl;
    }
    void loadModel()
    {
        Assimp::Importer importer;
        const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);
        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
        {
            cout << "ASSIMP:: !! ERROR !! -> " << importer.GetErrorString() << endl;
            return;
        }
        directory = path.substr(0, path.find_last_of('/'));
        processNode(scene->mRootNode, scene);

        model_loaded = true;
    }

    void processNode(aiNode* node, const aiScene* scene)
    {
        for (unsigned int i = 0; i < node->mNumMeshes; i++)
        {
            aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
            processVecMesh(mesh, scene);
        }
        for (unsigned int i = 0; i < node->mNumChildren; i++)
            processNode(node->mChildren[i], scene);
    }    

    void processVecMesh(aiMesh* mesh, const aiScene* scene)
    {
        for (unsigned int i = 0; i < mesh->mNumVertices; i = i + 3)
        {
            Tri tri;

            tri.vertex0.x = mesh->mVertices[i].x;
            tri.vertex0.y = mesh->mVertices[i].y;
            tri.vertex0.z = mesh->mVertices[i].z;

            tri.vertex1.x = mesh->mVertices[i+1].x;
            tri.vertex1.y = mesh->mVertices[i+1].y;
            tri.vertex1.z = mesh->mVertices[i+1].z;

            tri.vertex2.x = mesh->mVertices[i+2].x;
            tri.vertex2.y = mesh->mVertices[i+2].y;
            tri.vertex2.z = mesh->mVertices[i+2].z;

            triangles.push_back(tri);
        }
        
        std::cout << std::endl << triangles.size() << std::endl;

        scale(2.0f);

        for (unsigned int i = 0; i < triangles.size(); i++)
        {
            triangles[i].vertex0 = glm::vec4(triangles[i].vertex0, 1.0f) * model;
            triangles[i].vertex1 = glm::vec4(triangles[i].vertex1, 1.0f) * model;
            triangles[i].vertex2 = glm::vec4(triangles[i].vertex2, 1.0f) * model;
        }

    }
};
#endif