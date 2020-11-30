#pragma once

#include <glm/glm.hpp>

struct ProbeRay
{
    explicit ProbeRay(glm::vec3 orig,
					  glm::vec3 dir,
					  int p_index,
					  int l_index) {
		origin = orig;
        direction = dir;
		probe_index = p_index;
        local_index = l_index;
	}

	alignas(16) glm::vec3 origin;
    alignas(16) glm::vec3 direction;
	alignas(4) int probe_index; // what probe is this ray being shot from?
    alignas(4) int local_index; // what index does this ray have within the probe?
};

/*class Probe
{
	
};
*/