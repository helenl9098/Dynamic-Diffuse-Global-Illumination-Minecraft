#pragma once

#include <glm/glm.hpp>

struct ProbeRay
{
    explicit ProbeRay(glm::vec3 orig,
                      glm::vec3 dir,
					  int p_index) {
		origin = orig;
        direction = dir;
        probe_index = p_index;
	}

	alignas(16) glm::vec3 origin;
    alignas(16) glm::vec3 direction;
    alignas(4)  int probe_index;  // what probe is this ray being shot from?
};

/*class Probe
{
	
};
*/