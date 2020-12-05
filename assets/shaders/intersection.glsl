/*--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------*/
/*                                                                          */
/*                                                                          */
/*                             INTERSECTION					                */
/*                   									                    */
/*                                                                          */
/*--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------*/

/*
	TODO:
	
	- Fix triangles to use vec3 for vertices and not vec4.
	- Profile if const in and so on is faster.
	- Profile whether intersect_any vs intersect is slower/faster.
	- Add uvs for texturing.
	- Use mat4 or mat3 + vec3 for sphere data (allows ellipsoids).
	- Bring this further to instancing.
	- Add material data from intersection.
	
	- Add aabb intersection.
	- Add torus intersection.
	- Add cylinder intersection.
	- Add disk intersection.
	- Add quadrics intersection.
	- Add splines/patches intersection.
	- Write derivations for all new introduced intersections in the 
	appendix.

*/

/*--------------------------------------------------------------------------*/

struct Material_new
{
	int  type; /* 0: diffuse, 1: perfect mirror */
	vec3 base_color;
	vec3 emissive;
    float ior;
};

Material_new convert_old_material

	(Material mat)
	
{
	Material_new res;
	res.type = int(mat.data.x);
	res.base_color = mat.albedo.xyz;
	res.emissive = mat.emission.xyz;
    res.ior = mat.albedo.w;
	
	return res;
}

struct Isect

/*
	Structure containing the resulting data from an intersection.
*/

{
	float t;      /* coordinate along the ray, INF -> none */
	vec3  pos;    /* position in global coordinates */
	vec3  normal; /* normal in global coordinates */
	vec2  uv;     /* surface parametrization (for textures) */
	Material_new mat;
	int type;     /* 0/1 if no intersection,
					 2   if it's a light
					 3   if it's any block */
}; /* Isect */

/*--------------------------------------------------------------------------*/

bool intersect_sphere_any

	(Ray   ray,  /* ray for the intersection */
	 float mint, /* lower bound for t */
	 float maxt) /* upper bound for t */
	 
/*
	Returns true if there is an intersection with the unit sphere 
	with origin (0,0,0). The intersection is accepted if it is in 
	(mint, maxt) along the ray.
	
	For the derivation see the appendix.
*/
	 
{
	/* A*t^2 - 2*B*t + C = 0 */
	float A = dot(ray.direction, ray.direction);
	float B = -dot(ray.direction, ray.origin);
	float C = dot(ray.origin, ray.origin) - 1;
	
	/* discriminant */
	float D = B*B-A*C;
	D = D>0 ? sqrt(D) : INF;
	
	/* compute the two roots */
	float t1 = (B-D)/A;
	float t2 = (B+D)/A;
	
	/* check bounds validity in (mint, maxt) */
	t1 = mint < t1 && t1 < maxt ? t1 : INF;
	t2 = mint < t2 && t2 < maxt ? t2 : INF;

	/* pick the closest valid root */
	return min(t1,t2)<INF;
	
} /* intersect_sphere_any */

/*--------------------------------------------------------------------------*/

bool intersect_sphere

	(Ray       ray,  /* ray for the intersection */
	 float     mint, /* lower bound for t */
	 float     maxt, /* upper bound for t */
	 out Isect info) /* intersection data */
	 
/*
	Returns true if there is an intersection with the unit sphere 
	with origin (0,0,0). The intersection is accepted if it is in 
	(mint, maxt) along the ray. Additionally, the intersection 
	position and normal are computed (the direction of the normal is 
	outside facing).
	
	For the derivation see the appendix.
*/
	 
{

	/* A*t^2 - 2*B*t + C = 0 */
	float A = dot(ray.direction, ray.direction);
	float B = -dot(ray.direction, ray.origin);
	float C = dot(ray.origin, ray.origin) - 1;
	
	/* discriminant */
	float D = B*B-A*C;
	D = D>0 ? sqrt(D) : INF;
	
	/* compute the two roots */
	float t1 = (B-D)/A;
	float t2 = (B+D)/A;
	
	/* check bounds validity in (mint, maxt) */
	t1 = mint < t1 && t1 < maxt ? t1 : INF;
	t2 = mint < t2 && t2 < maxt ? t2 : INF;

	/* compute intersection data */
	info.t = min(t1,t2);
	info.pos = ray.origin + info.t*ray.direction;
	info.normal = info.pos;
	info.uv = vec2(0);
	
	return info.t<INF;
	
} /* intersect_sphere  */

/*--------------------------------------------------------------------------*/

bool intersect_plane_any

	(Ray   ray,  /* ray for the intersection */
	 float d,    /* offset of the plane: <o,n>, o any point on the plane */
	 vec3  n,    /* normal of the plane (not necessarily unit length) */
	 float mint, /* lower bound for t */
	 float maxt) /* upper bound for t */
	
/*
	Returns true if there is an intersection with the plane with the 
	equation: <p,n> = d. The intersection is accepted if it is in 
	(mint, maxt) along the ray.
	
	For the derivation see the appendix.
*/
	
{
	float t = (d-dot(ray.origin, n)) / dot(ray.direction,n);
	return mint < t && t < maxt;
	
} /* intersect_plane_any */

/*--------------------------------------------------------------------------*/

bool intersect_plane

	(Ray       ray,  /* ray for the intersection */
	 float     d,    /* offset of the plane: <o,n>, o: point on plane */
	 vec3      n,    /* normal of the plane (not necessarily unit length) */
	 float     mint, /* lower bound for t */
	 float     maxt, /* upper bound for t */
	 out Isect info) /* intersection data */
	
/*
	Returns true if there is an intersection with the plane with the 
	equation: <p,n> = d. The intersection is accepted if it is in 
	(mint, maxt) along the ray.
	
	Also computes the normal along the ray.
	
	For the derivation see the appendix.
*/
	
{
	float t = (d-dot(ray.origin,n)) / dot(ray.direction,n);
	
	bool isect = mint < t && t < maxt;
	
	info.t = isect ? t : INF;
	info.normal = normalize(n);
	
	return isect;
	
} /* intersect_plane */

/*--------------------------------------------------------------------------*/

bool intersect_triangle_any

	(Ray   ray,  /* ray for the intersection */
	 vec3  v0,   /* vertex 0 */
	 vec3  v1,   /* vertex 1 */
	 vec3  v2,   /* vertex 2 */
	 float mint, /* lower bound for t */
	 float maxt) /* upper bound for t */
	 
/*
	Returns true if there is an intersection with the triangle (v0,v1,v2).
	The intersection is accepted if it is in (mint, maxt) along the ray.
	Uses 3x3 matrix inversion for intersection computation.
	
	For the derivation see the appendix.
*/

{
	/* linear system matrix */
	mat3 A = mat3(ray.direction, v1-v0, v2-v0);
	
	/* formal solution A * x = b -> x = A^{-1} * b */
	vec3 sol =  inverse(A) * (ray.origin - v0);
	
	/* need to flip t, since the solution actually computes -t */
	float t = -sol.x;
	
	/* barycentric coordinates */
	float u = sol.y;
	float v = sol.z;

	return mint<t && t<maxt && 0<u && 0<v && u+v<1;
	
} /* intersect_triangle_any */

/*--------------------------------------------------------------------------*/

bool intersect_triangle

	(Ray       ray,  /* ray for the intersection */
	 vec3      v0,   /* vertex 0 */
	 vec3      v1,   /* vertex 1 */
	 vec3      v2,   /* vertex 2 */
	 float     mint, /* lower bound for t */
	 float     maxt, /* upper bound for t */
	 out Isect info) /* intersection data */

/*
	Returns true if there is an intersection with the triangle (v0,v1,v2).
	The intersection is accepted if it is in (mint, maxt) along the ray.
	Uses 3x3 matrix inversion for intersection computation.
	
	Computes also the intersection position, normal, uv coordinates.
	
	For the derivation see the appendix.
*/

{
	/* linear system matrix */
	mat3 A = mat3(ray.direction, v1-v0, v2-v0);
	
	/* formal solution A * x = b -> x = A^{-1} * b */
	vec3 sol =  inverse(A) * (ray.origin - v0);
	
	/* need to flip t, since the solution actually computes -t */
	float t = -sol.x;
	
	/* barycentric coordinates */
	float u = sol.y;
	float v = sol.z;
	
	/* is the intersection valid? */
	bool isect = mint<t && t<maxt && 0<u && 0<v && u+v<1;
	
	/* compute intersection data */
	info.t = isect ? sol.x : INF;
	info.pos = ray.origin + info.t*ray.direction;
	info.normal = normalize(cross(A[1],A[2]));
	info.uv = vec2(u,v);
	
	return isect;
	
} /* intersect_triangle */

/*--------------------------------------------------------------------------*/

bool intersect_triangle_any_fast

	(Ray   ray,  /* ray for the intersection */
	 vec3  v0,   /* vertex 0 */
	 vec3  v1,   /* vertex 1 */
	 vec3  v2,   /* vertex 2 */
	 float mint, /* lower bound for t */
	 float maxt) /* upper bound for t */
	 
/*
	Returns true if there is an intersection with the triangle (v0,v1,v2).
	The intersection is accepted if it is in (mint, maxt) along the ray.
	Uses the metric tensor for intersection computation.
	
	For the derivation see the appendix.
*/

{	
	/* edges and non-normalized normal */
	vec3 e0 = v1-v0;
	vec3 e1 = v2-v0;
	vec3 n = cross(e0,e1);
	
	/* intersect plane in which the triangle is situated */
	float t = dot(v0-ray.origin,n) / dot(ray.direction,n);
	vec3 p = ray.origin + t*ray.direction;
	
	/* intersection position relative to v0 */
	vec3 p0 = p - v0;
	
	/* transform p0 with the basis vectors */
	vec2 b = vec2(dot(p0,e0), dot(p0,e1));
	
	/* adjoint of the 2x2 metric tensor (contravariant) */
	mat2 A_adj = mat2(dot(e1,e1), -dot(e0,e1), -dot(e0,e1), dot(e0,e0));
	
	/* denominator of the inverse 2x2 metric tensor (contravariant) */
	float inv_det = 1.0/(A_adj[0][0]*A_adj[1][1]-A_adj[0][1]*A_adj[1][0]);
	
	/* barycentric coordinate */
	vec2 uv = inv_det * (A_adj * b);
	
	return mint<t && t<maxt && 0<uv.x && 0<uv.y && uv.x+uv.y<1;
	
} /* intersect_triangle_any_fast */

/*--------------------------------------------------------------------------*/

bool intersect_triangle_fast

	(Ray       ray,  /* ray for the intersection */
	 vec3      v0,   /* vertex 0 */
	 vec3      v1,   /* vertex 1 */
	 vec3      v2,   /* vertex 2 */
	 float     mint, /* lower bound for t */
	 float     maxt, /* upper bound for t */
	 out Isect info) /* intersection data */

/*
	Returns true if there is an intersection with the triangle (v0,v1,v2).
	The intersection is accepted if it is in (mint, maxt) along the ray.
	Uses the metric tensor for intersection computation.
	
	Computes also the intersection position, normal, uv coordinates.
	
	For the derivation see the appendix.
*/
	 
{
	/* edges and non-normalized normal */
	vec3 e0 = v1-v0;
	vec3 e1 = v2-v0;
	vec3 n = cross(e0,e1);
	
	/* intersect plane in which the triangle is situated */
	float t = dot(v0-ray.origin,n) / dot(ray.direction,n);
	vec3 p = ray.origin + t*ray.direction;
	
	/* intersection position relative to v0 */
	vec3 p0 = p - v0;
	
	/* transform p0 with the basis vectors */
	vec2 b = vec2(dot(p0,e0), dot(p0,e1));
	
	/* adjoint of the 2x2 metric tensor (contravariant) */
	mat2 A_adj = mat2(dot(e1,e1), -dot(e0,e1), -dot(e0,e1), dot(e0,e0));
	
	/* denominator of the inverse 2x2 metric tensor (contravariant) */
	float inv_det = 1.0/(A_adj[0][0]*A_adj[1][1]-A_adj[0][1]*A_adj[1][0]);
	
	/* barycentric coordinate */
	vec2 uv = inv_det * (A_adj * b);
	
	/* is it a valid intersection? */
	bool isect = mint<t && t<maxt && 0<uv.x && 0<uv.y && uv.x+uv.y<1;
	
	/* compute intersection data */
	info.t = isect ? t : INF;
	info.pos = ray.origin + info.t*ray.direction;
	info.normal = n;
	info.uv = uv;
	
	return isect;
	
} /* intersect_triangle_fast */

/*--------------------------------------------------------------------------*/

bool intersect_scene_any

	(Ray   ray,  /* ray for the intersection */
	 float mint, /* lower bound for t */
	 float maxt) /* upper bound for t */
	 
/*
	Returns true if there is an intersection with any primitive in the 
	scene in (mint, maxt). 
	
	Note: this intersection routine has an early out through an `if`, idk 
	if it does more bad than good in the sense of divergence, but 
	it allows skipping any check after the first intersection. 
*/
	 
{
	/* intersect spheres */
	for (int i = 0; i < spheres.length(); i++)
	{
		Sphere sphere = spheres[i];
		
		/* inverse transform on the ray, needs to be changed to 3x3/4x4 mat */
		Ray temp_ray;
		temp_ray.origin = (ray.origin - sphere.origin)/sphere.radius;
		temp_ray.direction = ray.direction / sphere.radius;
		
		/* early out */
		if (intersect_sphere_any(temp_ray, mint, maxt))
			return true;
	}
	
	/* intersect triangles */
	for (int i = 0; i < triangles.length(); i++)
	{
		Triangle tri = triangles[i];
		
		/* early out */
		if (intersect_triangle_any(ray, 
								   tri.vert0.xyz, 
								   tri.vert1.xyz, 
								   tri.vert2.xyz, 
								   mint, maxt)) return true;
	}
	
	return false;
	
} /* intersect_scene_any */


float sdBox(vec3 p, vec3 b)
{
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

float sdSphere(vec3 p, float s)
{
  return length(p)-s;
}

float opUnion( float d1, float d2 ) {  return min(d1,d2); }
float opSubtraction( float d1, float d2 ) { return max(-d1,d2); }
 
float opRep(in vec3 p, in vec3 c)
{
    vec3 q = mod(p+0.5*c,c)-0.5*c;
    return sdSphere(q, 0.05);
}


float opRepLim( in vec3 p, in float c, in vec3 l, out vec3 probePos)
{
	vec3 probeOrigin = c*clamp(round(p/c),-l,l);
	//probeOrigin += field_origin;
	
	probePos = probeOrigin;
	
    vec3 q = p-probeOrigin;
    return sdSphere(q, 0.5); // probe radius here
}

float sceneSDF(vec3 point, ivec3 probeCount, float sideLength, out vec3 probePos, vec3 field_origin) {

	return opRepLim(point - field_origin, sideLength, vec3(probeCount / 2), probePos);
}

vec3 estimateNormal(vec3 pos, ivec3 probeCount, float sideLength) {

    float epsilon = 0.0001;
    vec3 normal = vec3(0);

	vec3 temp = vec3(0);
    normal.x = sceneSDF(vec3(pos.x + epsilon, pos.y, pos.z), probeCount, sideLength, temp, vec3(0))
              - sceneSDF(vec3(pos.x - epsilon, pos.y, pos.z), probeCount, sideLength, temp, vec3(0));
    normal.y = sceneSDF(vec3(pos.x, pos.y + epsilon, pos.z), probeCount, sideLength, temp, vec3(0))
              - sceneSDF(vec3(pos.x, pos.y - epsilon, pos.z), probeCount, sideLength, temp, vec3(0));
    normal.z = sceneSDF(vec3(pos.x, pos.y, pos.z + epsilon), probeCount, sideLength, temp, vec3(0))
              - sceneSDF(vec3(pos.x, pos.y, pos.z - epsilon), probeCount, sideLength, temp, vec3(0));

    return normalize(normal);
}


// this is what ray traces the probes
bool implicit_surface(Ray ray, float mint, float maxt, ivec3 probeCount,
					  float sideLength, out Isect info, out vec3 probePos,
					  vec3 field_origin) {

// start of signed distance
	vec3 ray_origin = ray.origin;
	vec3 curr_cell = vec3(floor(ray.origin));
	vec3 ray_dir = normalize(ray.direction); 

	float curr_t = 0.f;
	bool isec = false;
	while (curr_t < (100)) {

		vec3 point = ray_origin + curr_t * ray_dir;
		float dist = sceneSDF(point, probeCount, sideLength, probePos, field_origin);

		if (dist < 0.001) {
			info.t = curr_t;
            info.normal = estimateNormal(point, probeCount, sideLength);
           	return true;
		}
		curr_t += dist;
	}
	return false;
//end of signed distance

}

float cave_sdf(vec3 coords) {
	float sdf = sdSphere(coords, 10.0);
	return opUnion(sdf, sdSphere(coords + vec3(2, 2, 2), 5.0));
}

float random1( vec3 p ) {
    return fract(sin((dot(p, vec3(127.1,
                                  311.7,
                                  191.999)))) *         
                 43758.5453);
}

float noise2D( vec2 p ) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) *
                 43758.5453);
}


float interpNoise2D(float x, float y) {
    int intX = int(floor(x));
    float fractX = fract(x);
    int intY = int(floor(y));
    float fractY = fract(y);

    float v1 = noise2D(vec2(intX, intY));
    float v2 = noise2D(vec2(intX + 1, intY));
    float v3 = noise2D(vec2(intX, intY + 1));
    float v4 = noise2D(vec2(intX + 1, intY + 1));

    float i1 = mix(v1, v2, fractX);
    float i2 = mix(v3, v4, fractX);
    return mix(i1, i2, fractY);
}


float fbm(float x, float y) {
    float total = 0;
    float persistence = 0.5;
    int octaves = 8;

    for(int i = 1; i <= octaves; i++) {
        float freq = pow(2.f, i);
        float amp = pow(persistence, i);

        total += interpNoise2D(x * freq,
                               y * freq) * amp;
    }
    return total;
}


int getBlockAt(vec3 coords, int scene) {

	/* BLOCK TYPE KEY: 
	0: EMPTY
	1: NOISE
	2: RED
	3. GREEN
	4. BLUE
	5. WHITE
	*/

	// TO DO: STUB FOR Now

	if (scene == 0) {// THIS IS THE CAVE
		if (coords.y > 17.0) {
			return 0;
		}
		if (coords.y < -15) {
			float r = fbm(coords.x * 0.1, coords.z * 0.1);
			int d = int(floor(r * 4.0));
			if (-19 + d >= coords.y) {
				return 1;
			}
		}
		if (sdSphere(coords, 20.0) > 0.0) {
			if (sdSphere(coords + vec3(16, 8, -10), 20.0) > 0.0) {
				if (sdSphere(coords + vec3(-13, -1, 19), 18.0) > 0.0) {
					if (sdSphere(coords + vec3(-6, -5, -4), 8.0) > 0.0) {
						if (sdSphere(coords + vec3(-18, -10, 24), 10.0) > 0.0) {
							if (sdSphere(coords + vec3(20, 15, 15), 21.0) > 0.0) {
									return 1;
							}
						}
					}
				}
			}
		}
	}

	else if (scene == 1) { // CORNELL BOX SCENE
		// x walls (left )
		if (coords.x == -10) {
			if (abs(coords.y) < 10 && abs(coords.z - 15) < 10) {
				return 2;
			} 
		}
	     // x walls ( right)
		if (coords.x == 10) {
			if (abs(coords.y) < 10 && abs(coords.z - 15) < 10) {
				return 3;
			} 
		}
		// y walls (ceiling floor)
		if (abs(coords.y) == 10) {
			if (abs(coords.x) < 10 && abs(coords.z - 15) < 10) {
				return 5;
			} 
		}
		// z wall (back)
		if (coords.z == 25) {
			if (abs(coords.x) < 10 && abs(coords.y) < 10) {
				return 5;
			} 
		}

		if (abs(coords.x + 3) < 3 && abs(coords.y + 7) < 3 && abs(coords.z -13) < 3) {
				return 4;
		}

		if (abs(coords.x - 4) < 3 && abs(coords.y + 4) < 6 && abs(coords.z -16) < 3) {
				return 4;
		} 
	}

	else if (scene == 2) { // HOUSE SCENE
		if (coords.y == -5) {
			return 1;
		}
		if (abs(coords.x) == 25) {
			if (abs(coords.y) < 5 && abs(coords.z) < 15) {
				return 2;
			}
		}
		if (coords.y == 5) {
			if (abs(coords.x) < 25 && abs(coords.z) < 15) {
				return 5;
			} 
		}
		if (coords.z == -15) {
			if (abs(coords.x) < 25 && abs(coords.y) < 5) {
				return 3;
			} 
		}
		if (coords.z == 15) {
			if (abs(coords.x - 10) < 2 && abs(coords.y + 1) < 4) {
				return 0;
			}
			if (abs(coords.x) < 25 && abs(coords.y) < 5) {
				return 3;
			} 
		}
	}
	else {
		return 0;
	}
	
	return 0;
}

vec2 getUVs(vec3 point, vec3 normal) {
	vec2 result_uv = vec2(0, 0);

	// unless the normal is straight up and down, the u in uvs is always x or z
	if (normal[1] == 0) {
		if (normal[0] == 0) {
			// this means the face is pointing in the z direction (forwards and back)
			if (sign(normal[2]) > 0) {
				result_uv[0] = ceil(point[0]) - point[0];
				result_uv[1] = point[1] - floor(point[1]);
			} else { 
				result_uv[0] = point[0] - floor(point[0]);
				result_uv[1] = point[1] - floor(point[1]);
			}
		}
		else { // this means the face is pointing in the x direction
			if (sign(normal[0]) < 1) {
				result_uv[0] = ceil(point[2]) - point[2];
				result_uv[1] = point[1] - floor(point[1]);
			} else {
				result_uv[0] = point[2] - floor(point[2]);
				result_uv[1] = point[1] - floor(point[1]);
			}
		}
	}
	else { // else, the us are X and the vs are Z
		if (sign(normal[1]) < 0) {
			result_uv[0] = point[0] - floor(point[0]);
			result_uv[1] = ceil(point[2]) - point[2];
		} else {
			result_uv[0] = point[0] - floor(point[0]);
			result_uv[1] = point[2] - floor(point[2]);
		}
	}
	return result_uv;
}

vec4 getColorAt(vec3 point, int block_type, vec3 normal) {
	/* BLOCK TYPE KEY: 
	0: EMPTY
	1: NOISE
	2: RED
	3. GREEN
	4. BLUE
	5. WHITE
	*/
 	if (block_type == 1) {
		float r = (random1(ceil(point)) / 4) + 0.1; // range of 0.3 to 0.8
		return vec4(0.1, r, r, 1);
	}
	else if (block_type == 2) {
		return vec4(1, 0, 0, 1);
	}
	else if (block_type == 3) {
		return vec4(0, 1, 0, 1);
	}
	else if (block_type == 4) {
		return vec4(0, 0, 1, 1);
	}
	else if (block_type == 5) {
		//vec2 uvs = getUVs(point, normal);
		//return vec4(uvs, 1, 1);
		return vec4(1, 1, 1, 1);
	}
}

// marches along ray and checks for blocks at locations
// changed from code given in CIS460
bool grid_march(Ray ray, float mint, float maxt, out Isect info, int scene) {
	vec3 ray_origin = ray.origin;
	vec3 curr_cell = vec3(floor(ray.origin));
	vec3 ray_dir = normalize(ray.direction);

	vec3 t2;
	float curr_t = 0.0;
	for (int i = 0; i < 200; i++) {
	    // calculate distance to voxel boundary
        t2 = max((-fract(ray_origin))/ray_dir, (1.-fract(ray_origin))/ray_dir);
        // go to next voxel
        float min_val = min(min(t2.x, t2.y), t2.z) + 0.0001;
        curr_t += min_val;
        ray_origin = ray.origin + ray_dir * curr_t;
        // get voxel's center
        vec3 pi = ceil(ray_origin) - 0.5;

        int block_type = getBlockAt(ceil(ray_origin), scene);
        if (block_type > 0) {
        	info.t = curr_t;

			// normal calculation
        	vec3 diff = normalize(ray_origin - pi);
        	vec3 normal = vec3(0, 0, 0);
        	float max = 0.0;
        	for (int i = 0; i < 3; i++) {
        		if (abs(diff[i]) > max) {
        			max = abs(diff[i]);
        			normal = vec3(0);
        			normal[i] = sign(diff[i]) * 1; 
        		}
        	}

        	info.normal = normalize(normal);

        	Material mat = materials[1]; // TO DO: Don't hard code this
        	mat.albedo = getColorAt(ray_origin, block_type, normalize(normal));
			info.mat = convert_old_material(mat);

        	return true;
        }
    }

	return false;
}


bool intersect_probes (
	 Ray  ray,  /* ray for the intersection */
	 float     mint, /* lower bound for t */
	 float     maxt, /* upper bound for t */
	 ivec3 probeCount,
	 float sideLength,
	 out Isect info, /* intersection data */ 
	 out vec3 probePos,
	 vec3 field_origin
	 )
{
	float closest_t = INF;
	info.t = closest_t;
	info.pos = vec3(0);
	info.normal = vec3(0);
	Isect temp_isect;

	if (implicit_surface(ray, mint, maxt, probeCount, sideLength, temp_isect, probePos, field_origin)) {
		info = temp_isect;
		closest_t = info.t;

		info.normal = closest_t<INF? normalize(info.normal) : vec3(0);
					
		info.pos = closest_t<INF? ray.origin + info.t * ray.direction : vec3(0);

		info.pos += 0.001 * info.normal;
	
		return true;
	}
	return false;
}

vec3 get_light_pos_in_scene(int scene) {
	vec3 light_pos = vec3(0);
	if (scene == 0) {
		light_pos = vec3(4, 17.5, 8.5) /*+ spheres[0].origin*/; // COMMENT THIS OUT TO STOP SPHERER FROM MOVING
	}
	if (scene == 1) {
		light_pos = vec3(0, 8, 13);
	}
	if (scene == 2) {
		light_pos = vec3(5, 9.3, 36.5);
	}
	return light_pos;
}

ivec2 get_text_coord_from_probe_number(int probe_number) {

	int x_dim = irradiance_field.probe_count.x * irradiance_field.probe_count.z;

	if (probe_number < 0 || x_dim < 0) {
		return ivec2(-1, -1);
	}
	ivec2 result = ivec2(-1, -1);	
	result[0] = int(mod(probe_number, x_dim));
	result[1] = int(floor(probe_number / x_dim));

	if (result[1] >= irradiance_field.probe_count.y) {
		return ivec2(-1, -1);
	}
	return result * irradiance_field.sqrt_rays_per_probe;
}

vec3 sample_probe(int probe_number, Isect info, int texture_to_sample) {

    // 1. Find where in the texture to sample
	// this is the top left corner of the n * n square that 
	// represents th probe in the texture
	ivec2 top_corner_text_coords = get_text_coord_from_probe_number(probe_number);

	
	// from the looks of things, they use the isect point's normal as the direction to sample
	// on the probe 
	vec3 irradiance_dir = normalize(info.normal);

	// need to change irradiance direction into a texture coord (relative to top left corner)
    // TO DO: Find texture coord
    ivec2 relative_text_coords = ivec2(0, 0);
    // float z = 1 - (2 * sample.x);
    // x  = ((-1 * (z - 1)) / 2) * sqrt_num_rays
    relative_text_coords[0] =  int(((-1.0 * (irradiance_dir[2] - 1.0)) / 2.0) * irradiance_field.sqrt_rays_per_probe);


    //relative_text_coords[y]

	// once I find the irradiance direction texture coord I add it to the top corner
	ivec2 sample_text_coord = top_corner_text_coords + relative_text_coords;

	// now I sample the image from these coords
	// TO DO: Specify which probe image to sample
	if (texture_to_sample == 0) {
		vec3 result = imageLoad(probe_image_albedo, sample_text_coord).xyz;
	} 
	else if (texture_to_sample == 1) {
		vec3 result = imageLoad(probe_image_distances, sample_text_coord).xyz;
	}
	else if (texture_to_sample == 2) {
		vec3 result = imageLoad(probe_image_normals, sample_text_coord).xyz;
	}

	// return result;
	return vec3(probe_number / (irradiance_field.probe_count[0] * irradiance_field.probe_count[1] * irradiance_field.probe_count[2]), 0, 0);
}

/*--------------------------------------------------------------------------*/

bool intersect_scene

	(Ray       ray,  /* ray for the intersection */
	 float     mint, /* lower bound for t */
	 float     maxt, /* upper bound for t */
	 out Isect info /* intersection data */
	 )
	 
{
	int scene = render_settings.scene; /*LOOK SCENE: NEEDED TO CHANGE SCENES*/
	float closest_t = INF;
	info.t = closest_t;
	info.pos = vec3(0);
	info.normal = vec3(0);
	Isect temp_isect;

	/* intersect spheres */
	//for (int i = 0; i < spheres.length(); i++)
	//{
		//Sphere sphere = spheres[i];
		
		// inverse transform on the ray, needs to be changed to 3x3/4x4 mat 
		Ray temp_ray;
		temp_ray.origin = (ray.origin - get_light_pos_in_scene(scene)) / 0.5;
		temp_ray.direction = ray.direction / 0.5;
		
        
        //    g(x) = 0, x \in S
        //    M(x) \in M(S) -> g(M^{-1}(x)) = 0 -> x \in S
            
        intersect_sphere(temp_ray, mint, closest_t, temp_isect);
		if (temp_isect.t<closest_t)
		{
			info = temp_isect;
			Material mat = materials[0];
			info.mat = convert_old_material(mat);
			info.type = 2; 
		}
		closest_t = min(temp_isect.t, closest_t);
	//} 

	/* intersect light */


	/* intersect triangles */
	/*
	for (int i = 0; i < triangles.length(); i++)
	{
		Triangle triangle = triangles[i];
		intersect_triangle_fast(ray, 
								triangle.vert0.xyz, 
								triangle.vert1.xyz, 
								triangle.vert2.xyz, 
								mint, 
								closest_t, 
								temp_isect);
		if (temp_isect.t<closest_t)
		{
			info = temp_isect;
			Material mat = materials[int(triangle.mat_id.x)];
			info.mat = convert_old_material(mat);
		}
		closest_t = min(temp_isect.t, closest_t);
	}
	*/

	// CHANGED: added floor
	/*
	intersect_plane(ray, -0.5, vec3(0, 1, 0), mint, maxt, temp_isect);
	if (temp_isect.t<closest_t)
	{
		info = temp_isect;
		Material mat = materials[2]; // TO DO: Don't hard code this
		info.mat = convert_old_material(mat);
		closest_t = min(temp_isect.t, closest_t);
	} */

	if (grid_march(ray, mint, maxt, temp_isect, scene)) {
		if (temp_isect.t<closest_t)
		{
			info = temp_isect;
			closest_t = info.t;
			info.type = 3;
		}
	}

	//closest_t = min(temp_isect.t, closest_t);
	
	info.normal = closest_t<INF? normalize(info.normal) : vec3(0);
					
	info.pos = closest_t<INF? ray.origin + info.t * ray.direction : vec3(0);

	info.pos += 0.001 * info.normal;
	
	return closest_t<INF;
	
} /* intersect_scene */

/*--------------------------------------------------------------------------*/

bool intersect_face

	(Ray       ray,  /* ray for the intersection */
	 vec3      center,    /* offset of the plane: <o,n>, o: point on plane */
	 vec3      min_coord,
	 vec3      max_coord,
	 vec3      n,    /* normal of the plane (not necessarily unit length) */
	 float     mint, /* lower bound for t */
	 float     maxt, /* upper bound for t */
	 out Isect info) /* intersection data */
	
/*
	Returns true if there is an intersection with the plane with the 
	equation: <p,n> = d. The intersection is accepted if it is in 
	(mint, maxt) along the ray.
	
	Also computes the normal along the ray.
	
	For the derivation see the appendix.
*/
	
{
	float denom = dot(ray.direction, n);
	if (abs(denom) > 0.0001) {
    	float t = dot((center - ray.origin), n) / denom;
    	bool isect = mint < t && t < maxt;
    	return isect;
    	/*
    	// if there is an intersection to the plane, we need 
	    // to see if its inside the face
		if (isect) {
			vec3 point = ray.origin + ray.direction * t;
			for (int i = 0; i < 3; i++) { // check if x y z in bounds
				if (point[i] < min_coord[i] || point[i] > max_coord[i]) {
					// oh no it's out of bounds
					return false;
				}
			}
		}
		info.t = t;
		info.normal = normalize(n);
		info.pos = ray.origin + ray.direction * t;
	
		return true;
		*/
	}
	return false;
	
} /* intersect_plane */


bool intersect_cube

	(Ray       ray,  /* ray for the intersection */
	 float     mint, /* lower bound for t */
	 float     maxt, /* upper bound for t */
	 vec3      coord,/* BOTTOM LEFT FRONT grid coord of cube */
	 out Isect info) /* intersection data */
	 
{
	float closest_t = INF;
	info.t = closest_t;
	info.pos = vec3(0);
	info.normal = vec3(0);
	Isect temp_isect;

	// need to test intersection on all faces
	// front
	if (intersect_face(ray, vec3(coord.x + 0.5, coord.y + 0.5, coord.z), // point on plane (did middle of face just for consistency)
							vec3(coord.x, coord.y, coord.z), /* min coord of face) */
							vec3(coord.x + 1, coord.y + 1, coord.z), /* max coord of face) */
							vec3(0, 0, 1), //normal,    /* normal of the plane (not necessarily unit length) */
	 						mint, maxt, temp_isect)) {
		if (temp_isect.t < closest_t) {
			info = temp_isect;
		}
	}
	
	info.normal = closest_t<INF? normalize(info.normal) : vec3(0);
					
	info.pos = closest_t<INF? ray.origin + info.t * ray.direction : vec3(0);
	
	return closest_t<INF;
	
} /* intersect_cube */

/*--------------------------------------------------------------------------*/

bool intersect_cubes_scene

	(Ray       ray,  /* ray for the intersection */
	 float     mint, /* lower bound for t */
	 float     maxt, /* upper bound for t */
	 out Isect info) /* intersection data */
	 
{
	float closest_t = INF;
	info.t = closest_t;
	info.pos = vec3(0);
	info.normal = vec3(0);
	Isect temp_isect;

	/* intersect spheres */
	for (int i = 0; i < spheres.length(); i++)
	{
		Sphere sphere = spheres[i];
		
		/* inverse transform on the ray, needs to be changed to 3x3/4x4 mat */
		Ray temp_ray;
		temp_ray.origin = (ray.origin - sphere.origin) / sphere.radius;
		temp_ray.direction = ray.direction / sphere.radius;
		
        /*
            g(x) = 0, x \in S
            M(x) \in M(S) -> g(M^{-1}(x)) = 0 -> x \in S
            
        */
        
		//intersect_sphere(temp_ray, mint, closest_t, temp_isect);
        intersect_sphere(temp_ray, mint, closest_t, temp_isect);
		if (temp_isect.t<closest_t)
		{
			info = temp_isect;
			Material mat = materials[int(sphere.mat_id.x)];
			info.mat = convert_old_material(mat);
		}
		closest_t = min(temp_isect.t, closest_t);
	}

	/*
	if (intersect_cube(ray, mint, maxt, vec3(0, 0, 0), temp_isect)) {
		if (temp_isect.t < closest_t)
		{
			info = temp_isect;
			Material mat = materials[1];
			info.mat = convert_old_material(mat);
			closest_t = temp_isect.t;
		}
	} */
	
	info.normal = closest_t<INF? normalize(info.normal) : vec3(0);
					
	info.pos = closest_t<INF? ray.origin + info.t * ray.direction : vec3(0);
	
	return closest_t<INF;
	
} /* intersect_scene */


vec3 get_sample(int probeIdx, vec3 dir, int textureTgt) { return vec3(0.f); }

vec3 get_diffuse_gi(Isect info, ivec3 probeCounts, int sideLength, Ray V)
{

	vec3 pos = info.pos;
    vec3 N = info.normal;
    V.direction = normalize(V.direction);	// view vector

	ivec3 baseProbeIdx = ivec3(floor(pos / float(sideLength)));

	ivec3 minProbeIdxIF = -(probeCounts / 2);

	vec3 sumIrradiance = vec3(0.f);
    float sumWeight = 0.f;

    vec3 alpha = (pos - baseProbeIdx * sideLength) / sideLength;

	for (int i = 0; i < 8; i++) {
        ivec3 offset = ivec3(i >> 2, i >> 1, i) & ivec3(1);
        vec3 probePos = vec3(round((baseProbeIdx + offset) * sideLength));
        ivec3 probeIdx3D = ivec3(probePos / float(sideLength)) - minProbeIdxIF;
        int probeIdx1D = probeIdx3D.x + probeIdx3D.z * probeCounts.x + probeIdx3D.y * probeCounts.x * probeCounts.z;

		vec3 dir = normalize(probePos - pos);

		vec3 trilinear = mix(1.0 - alpha, alpha, offset);

		// smooth backface test
		// all of these extra constants are supposed to prevent the weight
		// from going to zero
        float temp = max(0.0001, (dot(dir, N) + 1.0) * 0.5);
		// small addition term is supposed to prevent the weight from going to zero
        float weight = temp * temp + 0.2;

		// moment-visibility test
		// variance shadow map test
		// will need another texture to store the mean and teh mean squared
		// the author also linked a paper for that as well
        float isectProbeDist = length(pos - probePos);
		// sample form meanMeanSquared
        vec2 mms = get_sample(probeIdx1D, -dir, 0).rg;

        float mean = mms.x;
        float variance = abs(mean * mean - mms.y);

		temp = max(isectProbeDist - mean, 0.0);
        float chebyshevWeight = variance / (variance + temp * temp);

        // increase contrast in the weight
        chebyshevWeight = max(pow(chebyshevWeight, 3), 0.0);
		if (!(isectProbeDist <= mean))
        {
			weight *= chebyshevWeight;
		}

		// avoid zero weight
        weight = max(0.000001, weight);

		// sample from irradaince texture
		// this will also need to be made and evaluated using the other paper
		// that the author referenced
        vec3 irradiance = get_sample(probeIdx1D, N, 1).rgb;

		// amplifies dim lighting contributions to mimic the human visual
		// system's sensitivity to low light conditions
        const float crushThreshold = 0.2f;
        if (weight < crushThreshold)
        {
            weight *= weight * weight * (1.f / (crushThreshold * crushThreshold));
        }
        // scale by the trilinear weights
		// this scales the probe contribution such that probes that are far
		// away contribute the least
        weight *= trilinear.x * trilinear.y * trilinear.z;

		sumIrradiance += weight * irradiance;
        sumWeight += weight;
	}

	// combat the sensitive perception of very small amounts of light leaking
	// and then recursively lighting closed rooms by losing energy with each shade
	// this was also a uniform parameter in the supplemental code
	float energyPreservation = 0.98f;

	vec3 netIrradiance = energyPreservation * sumIrradiance / sumWeight;

    return 0.5f * PI * netIrradiance;
}




/*--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------*/
/*                                                                          */
/*                                                                          */
/*                          INTERSECTION APPENDIX			                */
/*                   									                    */
/*                                                                          */
/*--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------*/

/*
	Ray-Sphere intersection derivation:
	
	1) Ray in parametric form:
	r(t) = r.o + t*r.d
	
	2) Unit sphere centered at (0,0,0) canonical form:
	||p||^2 = 1^2
	
	3) For an intersecting point both 1) and 2) must hold.
	Substitute:
	p = r(t) = r.o + t*t.d ->
	
	||r.o + t*r.d||^2 = 1 ->
	
	||r.d||^2 * t^2 + 2*dot(r.o,r.d) * t + ||r.o||^2 - 1 = 0 ->
	
	A = dot(r.d, r.d), B = -dot(r.o,r.d), C = dot(r.o r.o) - 1 ->
	
	4) Quadratic equation:
	A * t^2 - 2*B * t + C = 0
	
	D = B*B - A*C, D<=0 -> no intersection
	(we ignore 1 point intersections since those have measure 0 
	and are irrelevant in an implementation with finite precision)
	
	5) If D>0 there are 2 intersections, we needs the closest 
	valid one from those. An intersection is valid if t is in (mint, maxt).
	
	The INF in the implementation guarantee that when there is 
	no intersection it will be rejected. It would have been nice if 
	sqrt(D) produced a NaN when D<0, but unfortunately it is undefined.
*/

/*--------------------------------------------------------------------------*/

/*

	Ray-Plane intersection derivation:
	
	1) Ray in parametric form:
	r(t) = r.o + t*r.d
	
	2) Plane in canonical form (cannot derive uv coords from it):
	dot(p,n) = d, if a point c is known on the plane then d = dot(o,n)
	
	n is the normal to the plane (doesn't have to be unit length, but 
	is preferable returning the correct normal without normalizing).
	
	3) For an intersection both 1) and 2) must hold.
	Substitute:
	p = r(t) = r.o + t*r.d ->
	
	dot(r.o+t*r.d,n) = d -> (using dot product linearity)
	
	t*dot(r.d, n) = d - dot(r.o, n) ->
	
	t = (d-dot(r.o,n)) / dot(r.d, n)
	
	4) The intersection is considered valid if t is in (mint, maxt).
	If t is NaN or +-Inf this check naturally fails.

*/

/*--------------------------------------------------------------------------*/

/*
	Ray-Triangle intersection derivation (3x3 matrix form):
	
	1) Ray in parametric form:
	r(t) = r.o + t*r.d
	
	2) Triangle in parametric form:
	S(u,v) = v0 + u*(v1-v0) + v*(v2-v0)
	
	v0,v1,v2 are the vertices' coordinates in global coordinates.
	
	3) For an intersection both 1) and 2) must hold.
	Equate:
	S(u,v) = r(t) ->
	
	v0 + u*(v1-v0) + v*(v2-v0) = r.o + t*r.d -> (group unknowns on rhs)
	
	-t*r.d + u*(v1-v0) + v*(v2-v0) = r.o - v0
	
	3) The above are 3 equations (for x,y,z coords) with 3 unknowns.
	Rewrite it matrix form for simplicity:
	
	A * sol = b, b = r.o - v0, sol = (-t, u, v), A = [r.d | v1-v0 | v2-v0],
	the input vectors to A are organized as column vectors.
	
	4) Formal solution: sol = inv(A) * b
	t = -sol[0], u = sol[1], v = sol[2]

	5) The intersection is valid, when:
	t in (mint, maxt) <- intersection with the plane of the triangle
	0<u and 0<v and u+v<1 <- the point's barycentric coordinates are 
	within the triangle.
	
	The above tests will fail naturally if the matrix is non-invertible 
	(there is no intersection) since the solution will be made of NaNs/Infs.
	
*/

/*--------------------------------------------------------------------------*/

/*
	Ray-Triangle intersection derivation (using the metric tensor):
	
	1) Ray in parametric form:
	r(t) = r.o + t*r.d
	
	2) Triangle in parametric form:
	e0 = v1-v0, e1 = v2-v0
	S(u,v) = v0 + u*e0 + v*e1
	
	v0,v1,v2 are the vertices' coordinates in global coordinates.
	
	3) Split the problem into a plane intersection + a subsequent 
	point-inside-triangle test.
	
	The plane in which the triangle lies is defined through the equation:
	dot(p-v0,n) = 0, n = cross(e0,e1)
	
	4) Ray-plane intersection:
	Substitute:
	p = r(t) = r.o + t*r.d ->
	dot(r.o+t*r.d-v0, n) = 0 -> (using linearity)
	t = dot(v0-r.o, n) / dot(r.d, n)
	
	Check t in (mint, maxt) -> fails naturally when 
	there is no intersection due to NaNs/Infs.
	
	5) Point-inside-triangle test. The ray-plane intersection point p 
	is inside the triangle if:
	p = S(u,v) is true for some (u,v) such that: 0<u and 0<v and u+v<1
	The above is the standard condition on the barycentric coordinate.
	
	The main issue is that p = S(u,v) is an overdetermined system with 
	3 equations and 2 unknowns which can result in edge cases. The 
	system can be reduced to a system of 2 equations.
	
	6) Metric tensor system reduction. The equation:
	p = S(u,v) = v0 + u*e0 + v*e1 can be dotted on both sides 
	with e0 and e1 ->
	
	dot(e0, p-v0) = u*dot(e0,e0) + v*dot(e0,e1)
	dot(e1, p-v0) = u*dot(e1,e0) + v*dot(e1,e1)
	
	This can be rewritten in matrix form:
	
	G * sol = b, b = (dot(e0,p-v0), dot(e1,p-v0)), sol = (u,v),
	
	    [ dot(e0,e0)   dot(e0,e1) ]
	G = [                         ] 
	    [ dot(e1,e0)   dot(e1,e1) ]
	
	G is the metric tensor of the basis {e0,e1}.
	
	7) System solution.
	Inversion of a 2x2 matrix is trivial:
	    [ a   b ]             [ d  -b ]
	A = [       ] -> inv(A) = [       ] * 1/(a*d-b*c)
	    [ c   d ]             [-c   a ]
		
	The solution is given as:
	sol = inv(G) * b, inv(G) is the contravariant metric tensor.
	
	7) If the matrix is non-invertible, then either e0=0 or e1 = 0
	or e0 and e1 are parallel, in which case the determinant is 0
	and results in NaNs/Infs, and the check 0<u && 0<v && u+v<1 
	naturally fails.
	
*/

/*--------------------------------------------------------------------------*/

bool intersect_spheres(Ray ray, inout Record record)
{
    float lowest = record.distance;
    for (int i = 0; i < spheres.length(); i++)
    {
        Sphere sphere = spheres[i];
        vec3 oc = ray.origin - sphere.origin;
        float b = dot(oc, ray.direction);
        float c = dot(oc, oc) - sphere.radius * sphere.radius;
        float delta = b * b - c;

        if (delta > 0.0f)
        {
            delta = sqrt(delta);
            float t0 = (-b - delta);
            float t1 = (-b + delta);
            float distance = min(t0, t1);
            if (!(t0 < 0 || t1 < 0) && RAY_MIN_DIST < distance && (distance < lowest || lowest < 0))
            {
                lowest = distance;
                record.hit = true;
                record.distance = distance;
                record.intersection = ray.origin + ray.direction * distance;
                record.normal = normalize(record.intersection - sphere.origin);
                record.mat = materials[int(sphere.mat_id.x)];
                record.albedo = materials[int(sphere.mat_id.x)].albedo.xyz;
                record.emission = materials[int(sphere.mat_id).x].emission.xyz;
            }
        }
    }
    return lowest > 0;
}


bool intersect_triangles(Ray ray, inout Record record)
{
    float lowest = record.distance;
    for(int i = 0; i < triangles.length(); i++){
        Triangle triangle = triangles[i];

        vec3 o = triangle.vert0.xyz;
        vec3 e0 = triangle.vert1.xyz - o;
        vec3 e1 = triangle.vert2.xyz - o;
        vec3 intersectionMat[3] = {ray.direction * -1, e0, e1};

        vec3 c01 = cross(intersectionMat[0], intersectionMat[1]);
        vec3 c12 = cross(intersectionMat[1],intersectionMat[2]);
        vec3 c20 = cross(intersectionMat[2], intersectionMat[0]);

        float det = dot(intersectionMat[0], c12);
        float inverseDet = 1.0f / det;

        vec3 inverseTransposedMat[3] = { c12*inverseDet, c20*inverseDet, c01*inverseDet };

        vec3 dir = ray.origin - o;
        vec3 tuv = vec3(
        dot(inverseTransposedMat[0], dir),
        dot(inverseTransposedMat[1], dir),
        dot(inverseTransposedMat[2], dir));

        if(0 < tuv.x && 0.0f < tuv.y && 0.0f < tuv.z && tuv.y + tuv.z < 1.0f){
            float t = tuv.x;
            if((RAY_MIN_DIST < t && (t < lowest || lowest < 0)))
            {
                lowest = t;
                record.intersection = ray.origin + ray.direction * t;
                record.distance = t;
                record.normal = vec3(triangle.vert0.w, triangle.vert1.w, triangle.vert2.w);
                record.hit = true;
                record.mat = materials[int(triangle.mat_id.x)];
                record.albedo = materials[int(triangle.mat_id.x)].albedo.xyz;
                record.emission = materials[int(triangle.mat_id).x].emission.xyz;
                //            rec.u = tuv.y * vertices[1].u + tuv.z * vertices[2].u + (1.0f - tuv.y - tuv.z) * vertices[0].u;
                //            rec.v = tuv.y * vertices[1].v + tuv.z * vertices[2].v + (1.0f - tuv.y - tuv.z) * vertices[0].v;
            }
        }
    }
    return lowest > 0;
}