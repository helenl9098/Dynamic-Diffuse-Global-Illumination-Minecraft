//
// Created by legend on 6/15/20.
//

#pragma once

#include <glm/glm.hpp>

struct Material
{
    enum class Type
    {
        LAMBERT,
        MIRROR,
        DIELECTRIC
    };
    explicit Material(glm::vec4 albedo, glm::vec4 emission, Type type)
        : albedo(albedo), emission(emission)
    {
        data = glm::vec4();
        data.x = (float)type;
    }
    glm::vec4 albedo{};
    glm::vec4 emission{};
    glm::vec4 data{};
};
