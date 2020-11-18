//
// Created by legend on 6/13/20.
//

#pragma once

struct Sphere
{
    explicit Sphere(glm::vec3 orig, float radiu, int mat_id){
        // No these aren't typos, I just couldn't think of anything else to name them...
        origin = orig;
        radius = radiu;
        material_id = glm::vec4(mat_id, 0, 0, 0);
    }
    glm::vec3 origin;
    float radius;
    glm::vec4 material_id;
    //glm::vec3 align;
};

struct Triangle
{
    explicit Triangle(glm::vec3 vert0, glm::vec3 vert1, glm::vec3 vert2, int mat_id)
    {
            glm::vec3 normal = glm::normalize(glm::cross(vert1 - vert0, vert2 - vert0));
            vertex0 = glm::vec4(vert0, normal.x);
            vertex1 = glm::vec4(vert1, normal.y);
            vertex2 = glm::vec4(vert2, normal.z);
            material_id = glm::vec4(mat_id, 0, 0, 0);
    }
    glm::vec4 vertex0{};
    glm::vec4 vertex1{};
    glm::vec4 vertex2{};
    glm::vec4 material_id{};
};
