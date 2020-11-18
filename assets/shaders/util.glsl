/*--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------*/
/*                                                                          */
/*                                                                          */
/*                                UTILITIES					                */
/*                   									                    */
/*                                                                          */
/*--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------*/

/*
	TODO:
	
	- Fix the PRNG.
*/

/*--------------------------------------------------------------------------*/

ivec2 image_size = imageSize(result_image);
//uint base_index = (gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * image_size.x) ^ 237283 * image_size.y;

uint index = 0;

/*
float rand () {
   return random_source[(base_index ^ ++index * 3927492) % random_source.length()];
}*/

    
uint wang_hash(uint seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

uint p_idx = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * image_size.x;
uint rng_state = wang_hash(p_idx)+iframe;

uint rand_xorshift()
{
    // Xorshift algorithm from George Marsaglia's paper
    rng_state ^= (rng_state << 13);
    rng_state ^= (rng_state >> 17);
    rng_state ^= (rng_state << 5);
    return rng_state;
}

float rand()
{
    return rand_xorshift() / 4294967296.0;
}
    
/*--------------------------------------------------------------------------*/

vec3 spherical_to_cartesian

	(float r,     /* radius */
	 float phi,   /* azimuth angle */ 
	 float theta) /* polar angle */

/*
	Computes the Cartesian coordinates from spherical coordinates.
	Uses the convention from physics and CG, not mathematics.
	
	x = r * cos(phi) * sin(theta)
	y = r * sin(phi) * sin(theta)
	z = r * cos(theta)
*/

{
	float sin_theta = sin(theta);
	return r*vec3(sin_theta * vec2(cos(phi), sin(phi)), cos(theta));
	
} /* spherical_to_cartesian*/

/*--------------------------------------------------------------------------*/

vec3 unit_spherical_to_cartesian

	(float phi,   /* azimuth angle */ 
	 float theta) /* polar angle */

/*
	Given the two angles (phi, theta) in spherical coordinates, computes
	the corresponding Cartesian coordinates on the unit sphere (r=1). 
	Uses the convention from physics and CG, not mathematics.
	
	x = cos(phi) * sin(theta)
	y = sin(phi) * sin(theta)
	z = cos(theta)
*/

{
	float sin_theta = sin(theta);
	return vec3(sin_theta * vec2(cos(phi), sin(phi)), cos(theta));
	
} /* unit_spherical_to_cartesian */

/*--------------------------------------------------------------------------*/

void construct_orthonormal_basis_Pixar

	(out vec3 e0,   /* x basis vector */
	 out vec3 e1,   /* y basis vector */
	 in  vec3 e2)   /* z basis vector */

/*
	Given a vector e2 generates an orthonormal basis: {e0, e1, e2}.
	
	Code from:
	Building an Orthonormal Basis, Revisited, (JCGT), vol. 6, no. 1, 1-8, 2017
	Available online: http://jcgt.org/published/0006/01/01/
*/

{
	float sign = e2.z < 0.0 ? -1.0 : 1.0;
	float a = -1.0 / (sign + e2.z);
	float b = e2.x * e2.y * a;
	e0 = vec3(1.0 + sign * e2.x * e2.x * a, sign * b, -sign * e2.x);
	e1 = vec3(b, sign + e2.y * e2.y * a, -e2.y);
	
	return;
	
} /* construct_orthonormal_basis_Pixar */

/*--------------------------------------------------------------------------*/

vec3 map_to_unit_hemisphere_around_normal

	(float phi,        /* azimuth angle */ 
	 float cos_theta,  /* cosine of the polar angle */
	 float sin_theta,  /* sine of the polar angle */
	 vec3  n)          /* unit normal */

/*
	Given the azimuthal and polar angles in spherical coordinates 
	returns the Cartesian coordinates of the point with these angles 
	on the unit hemisphere centered around the normal `n`.
	
	x = r * cos(phi) * sin(theta)
	y = r * sin(phi) * sin(theta)
	z = r * cos(theta)
*/

{

	float x = cos(phi) * sin_theta;
	float y = sin(phi) * sin_theta;
	float z = cos_theta;

	vec3 e0, e1;
	construct_orthonormal_basis_Pixar (e0, e1, n);
	
	return x * e0 + y * e1 + z * n;

} /* map_to_unit_hemisphere_around_normal */


/*--------------------------------------------------------------------------*/