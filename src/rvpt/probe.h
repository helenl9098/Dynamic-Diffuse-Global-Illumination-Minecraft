#pragma once

#include <glm/glm.hpp>

struct ProbeRay
{
    explicit ProbeRay(glm::vec3 orig,
                      glm::vec3 dir,
					  int p_index,
                      glm::vec2 t_offset) {
		origin = orig;
        direction = dir;
        probe_info = glm::vec3(p_index, t_offset.x, t_offset.y);
	}

	alignas(16) glm::vec3 origin;
    alignas(16) glm::vec3 direction;
    alignas(16) glm::vec3 probe_info;  // what probe is this ray being shot from,
                                       // and what is the texture offset?
};