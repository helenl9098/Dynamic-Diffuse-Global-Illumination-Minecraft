/*--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------*/
/*                                                                          */
/*                                                                          */
/*                         SAMPLES MAPPING					                */
/*                   									                    */
/*                                                                          */
/*--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------*/

/*
	TODO:
	
	- Add disk mapping.
	- Add disk -> hemisphere cosine mapping.
	- Add uniform triangle mapping.
*/

/*--------------------------------------------------------------------------*/

/*
	This should probably go in a separate file.
	
	A collection of functions for mapping samples in [0,1]^2
	to the sphere and hemisphere with different weighting 
	functions. In some PT implementations those are directly 
	associated with bsdfs, where a bsdf can sample itself.
	
	Simple variants for uniform and cosine sphere distributions 
	are provided which do not require the construction of an 
	orthonormal basis, which I believe should be faster in 
	general.
*/

/*--------------------------------------------------------------------------*/

vec3 map_uniform_sphere

	(float u,  /* x coordinate in [0,1], maps to azimuth angle */ 
	 float v)  /* y coordinate in [0,1], maps to polar angle */

/*
	Maps a point from [0,1]^2 to the unit sphere (boundary) with 
	uniform (constant) density.
	
	See the appendix for the derivation.
*/

{
	/* inverse transform sampling mapping */
	float phi = 2*PI*u;
	float cos_theta = 1-2*v;
	float sin_theta = sqrt(1-cos_theta*cos_theta);
	
	/* unit spherical to Cartesian coordinates */
	return vec3(sin_theta * vec2(cos(phi), sin(phi)), cos_theta);
	
} /* map_uniform_sphere */

/*--------------------------------------------------------------------------*/

vec3 map_uniform_hemisphere_simple

	(float u,  /* x coordinate in [0,1], maps to azimuth angle */ 
	 float v,  /* y coordinate in [0,1], maps to polar angle */
	 vec3  n)  /* unit normal */

/*
	Maps a point from [0,1]^2 to the unit sphere (boundary) with 
	uniform (constant) density.	Doesn't need to construct an 
	orthonormal basis.
*/

{
	vec3 p = map_uniform_sphere (u, v);
	return dot(n,p) < 0 ? -p : p;
	
} /* map_uniform_hemisphere_simple */

/*--------------------------------------------------------------------------*/

vec3 map_uniform_hemisphere

	(float u,  /* x coordinate in [0,1], maps to azimuth angle */ 
	 float v,  /* y coordinate in [0,1], maps to polar angle */
	 vec3  n)  /* unit normal */

/*
	Maps a point from [0,1]^2 to the unit hemisphere centered around 
	the normal `n` with uniform (constant) density.
	
	See the appendix for the derivation.
*/

{
	/* inverse transform sampling */
	float phi = 2*PI*u;
	float cos_theta = v;
	float sin_theta = sqrt(1-cos_theta*cos_theta);
	
	return map_to_unit_hemisphere_around_normal (phi, 
												 cos_theta, 
												 sin_theta, 
												 n);

} /* map_uniform_hemisphere */

/*--------------------------------------------------------------------------*/

vec3 map_cosine_hemisphere_simple

	(float u,  /* x coordinate in [0,1], maps to azimuth angle */ 
	 float v,  /* y coordinate in [0,1], maps to polar angle */
	 vec3  n)  /* unit normal */

/*
	Maps a point from [0,1]^2 to the unit hemisphere centered around 
	the normal `n` with cosine density. Doesn't need to construct an 
	orthonormal basis. Produces non-normalized points!
	
	For a geometric proof see: http://amietia.com/lambertnotangent.html
	For a calculus proof see: https://github.com/vchizhov/Derivations
*/

{
	/* offset a unit ball by the unit normal */
	return n + map_uniform_sphere(u,v);
	
} /* map_cosine_hemisphere_simple */

/*--------------------------------------------------------------------------*/

vec3 map_cosine_hemisphere

	(float u,  /* x coordinate in [0,1], maps to azimuth angle */ 
	 float v,  /* y coordinate in [0,1], maps to polar angle */
	 vec3  n)  /* unit normal */

/*
	Maps a point from [0,1]^2 to the unit hemisphere centered around 
	the normal `n` with cosine density. 
	
	See the appendix for the derivation.
*/

{
	/* inverse transform sampling */
	float phi = 2*PI*u;
	float cos_theta = sqrt(1-v);
	float sin_theta = sqrt(v);
	
	return map_to_unit_hemisphere_around_normal (phi,
												 cos_theta,
												 sin_theta,
												 n);
	
} /* map_cosine_hemisphere */

/*--------------------------------------------------------------------------*/

vec3 map_uniform_ball


	(float u,  /* x coordinate in [0,1], maps to azimuth angle */ 
	 float v,  /* y coordinate in [0,1], maps to polar angle */
	 float w)  /* z coordinate in [0,1], maps to radius */

/*
	Maps a point from [0,1]^3 to the unit ball with 
	uniform (constant) density.
	
	See the appendix for the derivation.
*/
	 
{
	/* inverse transform sampling */
	float phi = 2*PI*u;
	float cos_theta = 1-2*v;
	float sin_theta = sqrt(1-cos_theta*cos_theta);
	float r = pow(w, 1.0/3.0);
	
	/* map from spherical coordinates to Cartesian coordinates */
	return r * vec3(sin_theta * vec2(cos(phi), sin(phi)), cos_theta);
	
} /* map_uniform_ball */

/*--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------*/
/*                                                                          */
/*                                                                          */
/*                         SAMPLES MAPPING APPENDIX			                */
/*                   									                    */
/*                                                                          */
/*--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------*/

/*
	Algorithm for inverse transform sampling (1D):
	
	1) Supply a non-normalized density f.
	2) Normalize the density: 
		p = \frac{f}{\int_{a}^{b}f}
	3) Compute the CDF: 
		F(x) = \int_{a}^{x} p
	4) Invert the CDF: 
		x = F^{-1}(u), where u is uniformly distributed in [0,1]
		
	In 2D:
	3.1) Compute marginal density: 
		p_x (x) = \int_{y_0}^{y_1} p(x,y) \, dx
	3.2) Compute marginal CDF: 
		F_x(x) = \int_{x_0}^{x} p_x(t) \, dt
	3.3) Compute conditional density: 
		p_{y|x} = \frac{p_{x,y}}{p_x}
	3.4) Compute conditional CDF: 
		F_{y|x}(x,y) = \int_{y_0}^{y} \frac{p_{x,y}(x,t)}{p_x(x)} \, dt
	4.1) Invert marginal CDF:
		x = F_x^{-1}(u), u \in U[0,1]
	4.2) Invert conditional CDF:
		y = F_{y|x}^{-1}(x,v), v \in [0,1]
		
	Special case of independent variables:
	p_{x,y}(x,y) = p_x(x) * p_y(y) -> both can be sampled 
	independently from their own marginal distribution 
	(conditional is unnecessary).
*/

/*--------------------------------------------------------------------------*/

/*
	Notation:
	
	Unit ball:   B = \{p : ||p|| <= 1, p \in \mathbb{R}^3\}
	Unit sphere: S = \{p : ||p||  = 1, p \in \mathbb{R}^3\}, S = \partial B
	Unit hemisphere (centered around Z): H = \{p : p.z >= 0, p \in S\}

*/

/*--------------------------------------------------------------------------*/

/*
	Uniformly mapping points from [0,1]^2 to the unit sphere and unit 
	hemisphere:
	
	Unit sphere:     \phi \in [0,2\pi], \theta \in [0,\pi]
	Unit hemisphere: \phi \in [0,2\pi], \theta \in [0,0.5\pi]
	
	Denote A = S or A = H depending on whether sphere or 
	hemisphere is considered.
	
	1) Uniform mapping implies a constant density wrt the solid 
	angle measure (it is also constant in Cartesian coordinates 
	although multiplied with a Dirac delta).
	
	p_{\omega}(\omega) = C, p_{\cart} = C \delta_{A}
	p(\phi, \theta) = C |\sin\theta|, since 
	d\omega = |\sin\theta| d\phi d\theta
	
	2) Normalize the density:
	For S: \int_{S} p(\omega) d\omega = 1 -> C_S = \frac{1}{4\pi}
	For H: \int_{H} p(\omega) d\omega = 1 -> C_H = \frac{1}{2\pi}
	Let C = C_H, M_A = C_A / C
	
	3) Compute marginal densities:
	
	p_{\theta}(\theta) = \int_{0}^{2\pi}C_A \sin\theta d\phi 
					      = M_A\sin\theta
						  
	p_{\phi, \theta} = p_{\phi} * p_{\theta} ->
	p_{\phi}(\phi) = C
		
	4) Compute and invert CDFs:
	u,v uniform in [0,1]
	
	u = F_{\phi}(\phi) = \int_{0}^{\phi} p_{\phi}(t) dt ->
	\phi = 2\pi u
	
	v = F_{\theta}(\theta) = \int_{0}^{\theta} p_{\theta}(t) dt ->
	v = M_A ( 1 - \cos\theta) ->
	\cos\theta = 1 - v / M_A ->
	S: \cos\theta = 1-2v, H: \cos\theta = 1-v ~ v (due to v uniform in [0,1])
*/

/*--------------------------------------------------------------------------*/

/*
	Mapping points from [0,1]^2 to the cosine weighted hemisphere:
	
	1) Density of the form: p_{\omega}(\omega) = C |\cos\theta|
	
	2) Normalize density:
	\int_{0}^{2\pi}\int_{0}^{0.5\pi} C |\cos\theta| |\sin\theta| d\phi d\theta 
	= -2\pi C \int_{0}^{0.5\pi} \cos\theta d \cos\theta
	= -\pi C (cos^2(0.5\pi) - cos^2(0)) = C \pi = 1 -> C = 1 / \pi
	
	p(\phi,\theta) = \cos\theta |\sin\theta| / \pi
	
	3) Compute marginal densities:
	p_{\theta}(\theta) = \int_{0}^{2\pi}p(\phi,\theta) d\phi 
					   = 2 |\cos\theta| |\sin\theta|
	p_{\phi}(\phi) = p(\phi, \theta) / p_{\theta}(\theta) = \frac{1}{2\pi}
	
	4) Compute and invert CDFs:
	u = F_{\phi}(\phi) = \frac{\phi}{2\pi} -> \phi = 2\pi u
	v = F_{\theta}(\theta) = 1 - \cos^2 \theta -> \cos\theta = \sqrt{1-v}
	\sin\theta = \sqrt{1-\cos^2\theta} = \sqrt{v}
*/

/*--------------------------------------------------------------------------*/

/*

	Mapping points from [0,1]^3 to the uniform (constant) density 
	unit ball:
	
	1) Density of the form p_{x,y,z}(x,y,z) = C 1_{B}(x,y,z), where 
	1_{B} is the indicator function of the set B (the unit ball).
	
	p_{r,\phi,\theta}(r,\phi,\theta) = C r^2 |\sin\theta|,
	the non-constant part comes from the Jacobian of Spherical -> Cartesian.
	
	2) Normalize density:
	\int_{B} p = C \frac{4}{3} \pi = 1 -> C = \frac{3}{4\pi}
	
	3) Compute marginal densities:
	p_{r}(r) = \frac{r^2}{3}
	p_{\phi}(\phi) = \frac{1}{2\pi}
	p_{\theta}(\theta) = \frac{|\sin\theta|}{2}
	
	4) Compute and invert CDFs:
	\int_{(0,0,0)}^{r,\phi,\theta} p 
	= r^3 \frac{\phi}{2\pi} \frac{1-\cos\theta}{2}
	
	w = F_{r}(r) = r^3 -> r = \cbrt{z}
	u = F_{\phi}(\phi) = \frac{\phi}{2\pi} -> \phi = 2\pi u
	v = F_{\theta}(\theta) = \frac{1-\cos\theta}{2} -> \cos\theta = 1-2v

*/

/*--------------------------------------------------------------------------*/