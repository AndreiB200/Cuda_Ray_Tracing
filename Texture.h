#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "Math_Structs.cuh"

class Texture
{
public:
	unsigned int texName, scan;
	GLubyte image[16][16][16][4];

	unsigned int loadImage;

	void makeImage()
	{
		int s, t, r;
		for(s = 0; s < 16; s++)
			for (t = 0; t < 16; t++)
				for (r = 0; r < 16; r++)
				{
					image[r][t][s][0] = (s) * 17;
					image[r][t][s][1] = (t) * 17;
					image[r][t][s][2] = (r) * 17;
					image[r][t][s][3] = (r) * 17;
				}
	}

	void init()
	{
		makeImage();
		glGenTextures(1, &texName);
		glBindTexture(GL_TEXTURE_3D, texName);


		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

		glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA, 16, 16, 16, 0, GL_RGBA, GL_UNSIGNED_BYTE, image);

		glBindTexture(GL_TEXTURE_3D, 0);
	}

	void loadIm()
	{
		glGenTextures(1, &loadImage);
		glBindTexture(GL_TEXTURE_2D, loadImage);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		int x, y, channels;
		unsigned char* data = stbi_load("image/scan0/1.png", &x, &y, &channels, 0);

		if (data)
		{
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, x, y, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
			glGenerateMipmap(GL_TEXTURE_2D);
		}
		else
		{
			std::cout << "No loaded file" << std::endl;
			exit(0);
		}

		stbi_image_free(data);

		glBindTexture(GL_TEXTURE_2D, 0);
	}

	void loadScan()
	{
		glGenTextures(1, &scan);
		glBindTexture(GL_TEXTURE_3D, scan);

		auto wrap = GL_CLAMP_TO_BORDER;
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, wrap);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, wrap);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, wrap);

		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

		int x, y, channels;
		unsigned char* data = (unsigned char*)malloc(128 * 128 * 19 * 4);
		unsigned char* dataLoad;
		for (int i = 0; i < 19; i++)
		{
			std::string link = "image/scan0/"; std::string type = ".png";
			std::string number = std::to_string(i);
			std::string path = link + number + type;
			dataLoad = stbi_load(path.c_str(), &x, &y, &channels, 0);

			for (int j = x * y * channels * i; j < x * y * channels * (i + 1); j++)
			{
				data[j] = dataLoad[j - (x * y * channels * i)];
			}
			stbi_image_free(dataLoad);
		}

		glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA, x, y, 19, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
		glBindTexture(GL_TEXTURE_3D, 0);

		free(data);
	}
};