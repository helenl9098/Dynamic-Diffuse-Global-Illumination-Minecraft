/*--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------*/
/*                                                                          */
/*                                                                          */
/*                                MATERIALS					                */
/*                   									                    */
/*                                                                          */
/*--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------*/

/*
    TODO:
    
    - add fuzzy metal
    - add fuzzy dielectric
    - add Phong & modified Phong
    - add Cook-Torrance
    - add Oren-Nayar
    - add Textures
*/

/*--------------------------------------------------------------------------*/

vec3 handwritten_reflect

	(vec3 dir_in,  /* incident direction (dot(dir_in, normal)<=0) */
	 vec3 normal)  /* unit normal */
	
/*
	Reflects dir_in in the plane formed by dir_in and normal, around 
	normal (dot(dir_in, normal)<=0). The result is symmetric to -dir_in 
    wrt the normal. The normal vector is assumed to be unit length.

	This implementation is only for educational purposes, one can 
	use the builtin GLSL reflect without any issues.
	
	For the derivation see the appendix.
*/
	
{
	return dir_in - 2*dot(dir_in,normal)*normal;
 
} /* handwritten_reflect */

/*--------------------------------------------------------------------------*/

vec3 handwritten_refract

	(vec3  dir_in,  /* incident direction (dot(dir_in, normal)<=0) */
	 vec3  normal,  /* unit normal */
     float eta)     /* the ratio of the indices of refraction */
	
/*
	Refracts dir_in wrt the normal where the outside to inside ratio of 
    the refractive indices is equal to eta (dot(dir_in,normal)<=0). 
    It returns vec3(0) upon total internal reflection.

	This implementation is only for educational purposes, one can 
	use the builtin GLSL refract without any issues.
	
	For the derivation see the appendix.
*/
	
{
    float cos_a = dot(dir_in, normal);
    float cos_b_sqr = 1.0 - eta*eta * (1.0-cos_a*cos_a);
    if (cos_b_sqr<=0) /* total internal reflection */
        return vec3(0.0);
    else
        return eta*dir_in + (sqrt(max(0.0,cos_b_sqr))-eta*cos_a)*normal;
    
} /* handwritten_refract */

/*--------------------------------------------------------------------------*/

vec3 mat_eval_Lambert_cos

	(vec3 diffuse)  /* brdf constant (energy conserv.: diffuse<1/PI) */
    
/*
    Evaluation of the diffuse brdf*cos(theta) due to a cosine 
    weighted hemisphere sampling. The rendering equation's cosine 
    factor and the pdf's  cosine cancel out, and the pdf's PI 
    factor remains.
*/
	
{
	return diffuse * PI;
    
} /* mat_eval_Lambert_cos */

/*--------------------------------------------------------------------------*/

vec3 mat_scatter_Lambert_cos

	(vec3 normal)  /* unit normal */

/*
    Samples the Lambert bsdf * cos by using cosine weighted hemisphere 
    sampling.
*/
	 
{
	return map_cosine_hemisphere_simple(rand(), rand(), normal);
    
} /* mat_scatter_Lambert_cos */



/*--------------------------------------------------------------------------*/

vec3 mat_eval_Lambert

	(vec3 diffuse)
	
{
	return diffuse * PI;
}

/*--------------------------------------------------------------------------*/

vec3 mat_scatter_Lambert

	(vec3 normal)
	 
{
	vec3 dir = map_cosine_hemisphere_simple (rand(), rand(), normal);
	return dir;
}

/*--------------------------------------------------------------------------*/

vec3 mat_eval_mirror

	(vec3 tint)  /* brdf constant (energy conserv.: tint<1 */ 
    
/*
    Evaluates the perfect mirror brdf (it's actually a Dirac delta 
    so there's some logic leak to the integrators).
*/
	
{
	return tint;
    
} /* mat_eval_mirror */

/*--------------------------------------------------------------------------*/

vec3 mat_scatter_mirror


	(vec3 dir_in,  /* incident direction (dot(dir_in, normal)<=0) */
	 vec3 normal)  /* unit normal */
     
/*
    "Samples" the perfect mirror brdf (it's actually a Dirac delta 
    so there's some logic leak to the integrators).
*/
	 
{
	return reflect(dir_in, normal);
    
} /* mat_scatter_mirror */

/*--------------------------------------------------------------------------*/

vec3 mat_eval_dielectric

	(vec3 tint)  /* brdf constant (energy conserv.: tint<1 */ 
    
/*
    Evaluates the perfect dielectric brdf (it's actually a Dirac delta 
    so there's some logic leak to the integrators).
*/
	
{
	return tint;
    
} /* mat_eval_dielectric */

/*--------------------------------------------------------------------------*/

vec3 mat_scatter_dielectric

	(vec3 dir_in,  /* incident direction (dot(dir_in, normal)<=0) */
	 vec3 normal,  /* unit normal */
     float eta)    /* the ratio of the indices of refraction */
    
/*
    "Samples" the perfect dielectric brdf (it's actually a Dirac delta 
    so there's some logic leak to the integrators).
*/
	
{
	vec3 res = refract(dir_in, normal, eta);
    if (all(equal(res,vec3(0))))
        return reflect(dir_in, normal);
    else
        return res;
    
} /* mat_scatter_dielectric */

/*--------------------------------------------------------------------------*/

float frensel_reflectance

    (float cos_in,
     float cos_out,
     float eta)
     
/*
    Returns the percent of reflection at the boundary medium using 
    the Fresnel equations (for unpolarized light).
    The transmission is 1 minus the result.
    
    For further details see:
    Reflections and Refractions in Ray Tracing, Bram de Greve
*/
     
{
    float r_perp = (eta*cos_in - cos_out)/(eta*cos_in + cos_out);
    float r_parallel = (cos_in - eta*cos_out)/(cos_in + eta*cos_out);
    
    return 0.5*(r_perp*r_perp + r_parallel*r_parallel);
    
} /*  frensel_reflection */

/*--------------------------------------------------------------------------*/

vec3 handle_material

	(Material_new mat,
	 vec3 dir_in,
	 vec3 normal,
	 out vec3 dir_out)
	
{
	switch (mat.type)
	{
	case 0: /* Lambertian */
		dir_out = mat_scatter_Lambert_cos(normal);
		return mat_eval_Lambert_cos(mat.base_color);
	case 1: /* perfect mirror */
		dir_out = mat_scatter_mirror(dir_in, normal);
		return mat_eval_mirror(mat.base_color);
	case 2: /* dielectric */
        dir_out = mat_scatter_dielectric(dir_in, normal, 1.5);
		return mat_eval_dielectric(mat.base_color);

	default:
		dir_out = vec3(0);
		return vec3(0.0);
	}
}

/*--------------------------------------------------------------------------*/


/*--------------------------------------------------------------------------*/
/*                                                                          */
/*                                                                          */
/*                        MATERIALS	APPENDIX			                    */
/*                   									                    */
/*                                                                          */
/*--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------*/

/*
    Reflect derivation. 
    
    
    
    0) Given unit vectors d and n, such that dot(d,n) >= 0, 
    find r: r symmetric to d wrt to n.
    
            
    ------^------
   |\    /|\    /|
     \    |    /
      \  n|   /
    d  \  |  /  r
        \ | /
         \|/


    1) Decompose d in components parallel to n and perpendicular to n.
    The projection of d onto n: d_parallel = dot(d,n) * n
    The perpendicular part can be derived from: d = d_parallel + d_perp.
    
    d_perp
   <-------
   |\    /|\
     \    | d_parallel
   d  \   |
       \ /|\
        \ | n
         \|
         
    d_perp = d - d_parallel = d - dot(d,n) * n
    
    2) Represent r through d and d_perp:
    
   -d_perp -d_perp       
   ------->------>
   |\    /|\    /|
     \    |    /
   d  \   |   /  r
       \ /|\ /
        \ |n/
         \|/
        
    r = d - 2 * d_perp = d - 2*(d-dot(d,n)*n) = 2*dot(d,n) * n - d
    
    3) The convention in GLSL is that d is given as facing in the opposite 
    direction (dot(d,n)<=0), but r should be symmetric to -d wrt n. 
    For this it suffice to negate d: r = 2*dot(-d,n)*n-d = d-2*dot(d,n)*n
    
   -d_perp -d_perp       
   ------->------>
    \    /|\    /|
     \    |    /
   d  \   |   /  r
       \ /|\ /
        \ |n/
        _\|/
    

    The projection of d onto n, going in the same direction is:
    
    d_parallel = -dot(d,n) * n
    
    -d = d_parallel + d_perp
    
    d_perp = -d - d_parallel = -d + dot(d,n)*n
    
    r = -d -2*d_perp = -d -2*(-d + dot(d,n)*n) = d - 2*dot(d,n)*n
    
*/

/*--------------------------------------------------------------------------*/

/*
    Refract derivation.
    
    
    
    0) Let there be "outside" and "inside" media, respectively with indices 
    of refraction n1 and n2. Denote eta = n1/n2. Let the vector n be the 
    unit normal of the surface which points towards "outside", and d be a 
    unit vector such that dot(d,n) >= 0. Let angle(d,n) = a, angle(r,-n) = b.
    We derive how to find the refracted vector r.
     __
    |\     ^
      \   /|\       
       \ __|
      d \ a|
         \ | n
          \|         n1
    --------------------
           |\_       n2
           |__\_  r
        -n | b  \_
           |     _\|
          \|/
           v 

           
    1) Snell's law states: sin(a) * n1 = sin(b) * n2, 
    or equivalently (provided that n2 != 0) ->
    
    sin(b) = n1/n2 * sin(a)
    
    cos(a) = dot(n,d)
    sin(a) = sqrt(1-cos^(a)) = sqrt(1-dot(n,d)*dot(n,d))
    
    
    2) Total internal reflection happens when n1/n2 * sin^2(a) > 1, 
    since there exists no b for which sin(b) > 1. This is possible only 
    when n1/n2 > 1. In the case of total internal reflection, just 
    reflect the direction as for a mirror.
    
    3) If no internal reflection occurs, then:
    
    sin(b) = n1/n2 * sqrt(1-dot(n,d)*dot(n,d))
    cos(b) = sqrt(1-sin^2(b)) = sqrt(1-(1-dot(n,d)*dot(n,d))*(n1/n2)^2)
    
    The check: cos^(b) = k = 1 - (n1/n2)^2 * (1-dot(n,d)*dot(n,d)) < 0 
    can also be used to verify for total internal reflection, which 
    is used in the check used in GLSL's builtin refract function.
    
    
    4) The refracted ray can be reconstructed from a 2-vector basis:
    r = -n * cos(b) + n_perp * sin(b)
    
    d = d_parallel + d_perp = dot(d,n) * n + d_perp ->
    
    d_perp = d - dot(d,n) * n ->
    
    n_perp = -d_perp / ||d_perp|| = -d_perp / sin(a) ->
    
    r = -n * cos(b) - d_perp * (n1/n2) * sin(a) / sin(a)
      = -n * cos(b) - n1/n2 * d_perp
      = -n1/n2 * d + (n1/n2*dot(n,d) - cos(b)) * n
      
    5) The convention in GLSL is that d is reversed (dot(d,n)<=0), 
    that is d = -d' ->
    
    r = n1/n2 * d + (-cos(b)-n1/n2*dot(n,d)) * n
    
    Snell's law can be derived trivially from Fermat's principle 
    (least time path), as well from more complex considerations. 
    See the Wikipedia page for further details.
                  
    Implementation details:
        - make sure d and n are normalized
        - make sure that dot(d,n)>0 (4) or dot(d,n)<0 (5), if not, flip n
        - offset intersection for reflection by eps * n, and for refraction 
        by -eps*n, to avoid self-intersection
        - if `n` is used as an indicator of inside/outside, make the ior 
        reciprocal whenever n is flipped
          
*/

/*--------------------------------------------------------------------------*/

void apply_record(inout Ray ray, inout Record record)
{
    Material mat = record.mat;

    vec3 normal = record.normal;
    if(dot(normal, ray.direction) > 0) normal *= -1;

    if(mat.data.x == 0) // Lambert
    {
        ray.origin = record.intersection;
        float u = rand();
        float v = rand();
		ray.direction = normalize(map_cosine_hemisphere_simple(u, v, normal));
        record.reflectiveness = max(0, dot(normal, ray.direction));
    }
    else if (mat.data.x == 1) // Glass
    {
        ray.origin = record.intersection;
        float nit;
        vec3 normal = record.normal;
        if(dot(normal, ray.direction) > 0)
        {
            normal *= -1;
            nit = 1.5f;
        }
        else
        {
            nit = 1. / 1.5f;
        }
        ray.direction = refract(ray.direction, normal, nit);
        record.reflectiveness = 1.f;
    }
    else if (mat.data.x == 2) // Dynamic
    {
        ray.origin = record.intersection;
        ray.direction = reflect(ray.direction, normal); //normalize(random_unit_sphere_point()), mat.data.y);
        record.reflectiveness = 1.f;
    }
}
