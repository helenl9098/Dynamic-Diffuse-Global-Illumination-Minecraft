#version 450 core

layout(binding = 0) uniform Camera { mat4 matrix; }
cam;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_color;

layout(location = 0) out vec3 out_normal;
layout(location = 1) out vec3 out_color;

out gl_PerVertex { vec4 gl_Position; };

void main()
{
    gl_Position = cam.matrix * vec4(in_position, 1);
    /* flip image vertically */
    gl_Position.y = -gl_Position.y;
    out_normal = in_normal;
    out_color = in_color;
}