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
        probe_index = p_index;
        texture_offset = t_offset;
	}

	alignas(16) glm::vec3 origin;
    alignas(16) glm::vec3 direction;
    alignas(4)  int probe_index;  // what probe is this ray being shot from?
    alignas(8)  glm::vec2 texture_offset;
};