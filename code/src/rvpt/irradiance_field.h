#pragma once

#include "glm/glm.hpp"


class IrradianceField
{
public:
    IrradianceField();

    int numProbes();
    
    // specification
    glm::ivec3 probeCounts;
    int sideLength;
    float hysteresis;
    int raysPerProbe;
};