#pragma once

#include <glm/glm.hpp>

struct ProbeRay
{
    explicit ProbeRay(glm::vec3 orig, glm::vec3 dir) {
		origin = orig;
        direction = dir;
	}

	glm::vec3 origin;
    glm::vec3 direction;
};

/*class Probe
{
	
};
*/