#version 460 core

out vec4 FragColor;

 in vec3 position;
 in vec2 texCoord;
 in vec3 FragPos;

 in vec3 cameraPos;

 uniform sampler3D textures;
 uniform sampler2D tee;

 uniform float value;

void main()
{	
	
	vec3 coord = position + 0.5;
	vec3 reverse = vec3(coord.x,coord.y,coord.z);

	vec4 color = texture(tee, texCoord);
	
	vec4 changedColor = color - vec4(vec3(value), 1.0);
	
		
	FragColor = changedColor;
}