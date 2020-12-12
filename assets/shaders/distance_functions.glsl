/*--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------*/
/*                                                                          */
/*                                                                          */
/*                             DISTANCE FUNCTIONS					        */
/*                   									                    */
/*                                                                          */
/*--------------------------------------------------------------------------*/

struct IsectM

/*
    Marched intersection.
*/

{
    float t;
    float radius;
    int iter;
};

/*--------------------------------------------------------------------------*/

float distance_sphere

    (vec3 p)
    
/*
    Returns the signed distance from p to the surface of the unit sphere 
    centered at (0,0,0).
*/ 

{
    return length(p)-1.0;

} /* distance_sphere */

/*--------------------------------------------------------------------------*/

float dot2( in vec2 v ) { return dot(v,v); }
float dot2( in vec3 v ) { return dot(v,v); }

float distance_triangle

    (vec3 p,
     vec3 a,
     vec3 b,
     vec3 c)
     
/*
    Returns the distance from p to the triangle 
    formed by the vertices a,b,c.
    
    From: 
    https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
*/
    
{
  vec3 ba = b - a; vec3 pa = p - a;
  vec3 cb = c - b; vec3 pb = p - b;
  vec3 ac = a - c; vec3 pc = p - c;
  vec3 nor = cross( ba, ac );

  return sqrt(
    (sign(dot(cross(ba,nor),pa)) +
     sign(dot(cross(cb,nor),pb)) +
     sign(dot(cross(ac,nor),pc))<2.0)
     ?
     min( min(
     dot2(ba*clamp(dot(ba,pa)/dot2(ba),0.0,1.0)-pa),
     dot2(cb*clamp(dot(cb,pb)/dot2(cb),0.0,1.0)-pb) ),
     dot2(ac*clamp(dot(ac,pc)/dot2(ac),0.0,1.0)-pc) )
     :
     dot(nor,pa)*dot(nor,pa)/dot2(nor) );
     
} /* distance_triangle */

/*--------------------------------------------------------------------------*/

vec2 min_idx(vec2 lhs, vec2 rhs)
{
    return lhs.x<rhs.x ? lhs : rhs;
}

/*float intersect_scene_st

	(Ray        ray,  // ray for the intersection 
	 float      mint, // lower bound for t 
	 float      maxt, // upper bound for t 
	 out IsectM info) // intersection data 
     

    Intersect scene using sphere tracing.

     
{
    float t = mint;
    vec3 p = ray.origin + t*ray.direction;
    int i;
    vec2 s_radius_idx = vec2(INF,-1);
    vec2 t_radius_idx = vec2(INF,-1);
    float min_radius;
    for (i=0; i<MARCH_ITER; ++i)
    {
        for (int j=0; j<spheres.length(); ++j)
        {
            Sphere sphere = spheres[j];
            vec3 p_tform = (p - sphere.origin)/sphere.radius;
            float dist = sphere.radius * distance_sphere(p_tform);
            s_radius_idx = min_idx(s_radius_idx, vec2(dist, j));
        }
        
        t_radius_idx = vec2(INF, -1);
        for (int j=0; j<triangles.length(); ++j)
        {
            Triangle tri = triangles[j];
            float dist = distance_triangle(p, tri.vert0.xyz, tri.vert1.xyz, tri.vert2.xyz);
            t_radius_idx = min_idx(t_radius_idx, vec2(dist, j));
        }
        
        min_radius = min(s_radius_idx.x, t_radius_idx.x);
        if (min_radius < MARCH_EPS || min_radius > maxt)
        {
            info.t = t;
            info.radius = min_radius;
            info.iter = i;
            return t;
        }
        
        t += min_radius;
        p += min_radius * ray.direction;

    }
    info.t = INF;
    info.radius = min_radius;
    info.iter = i;
    return INF;
    
}*/ /* intersect_scene_st */

/*--------------------------------------------------------------------------*/

