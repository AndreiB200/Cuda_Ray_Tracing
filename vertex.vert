#version 460 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNor;
layout (location = 2) in vec2 aTex;

out vec3 position;
out vec3 FragPos;
out vec2 texCoord;

uniform mat4 proj;
uniform mat4 view;
uniform mat4 model;

void main()
{
	position = aPos;
	texCoord = aTex;
	FragPos = vec3(model * vec4(aPos,1.0));
	gl_Position = proj * view * model * vec4(aPos,1.0);
}