#pragma once

#include <iostream>
#include <string>
#include "Shader.h"
#include <GLFW/glfw3.h>
#include <glad/glad.h>

class ShaderAccess
{
public:
	float value = 0.0f;

	void changeValue(Shader& shader, GLFWwindow* window)
	{
		if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
		{
			value = value + 0.01f;
		}
		if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
		{
			value = value - 0.01f;
		}
		shader.setFloat("value", value);
	}
};