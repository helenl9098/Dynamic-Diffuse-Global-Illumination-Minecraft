#include "irradiance_field.h"

IrradianceField::IrradianceField() : 
	probeCounts(glm::ivec3(4, 4, 4)), 
	sideLength(4), 
	hysteresis(0.98f), 
	raysPerProbe(64)
{}

int IrradianceField::numProbes() {
	return probeCounts.x * probeCounts.y * probeCounts.z;
}