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


////////////////////////////////////////////////////////////////////////////////
// Helen Cube Marching

bool getBlockAt(vec3 coords) {

	// TO DO: STUB FOR Now
	if (length(coords) > 20.0 && coords.y < 18.0) {
		return true;
	}
	return false;
}

float sdBox(vec3 p, vec3 b)
{
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

float sdSphere(vec3 p, float s)
{
  return length(p)-s;
}

bool getVoxel(vec3 c) {
	vec3 p = c + vec3(0.5);
	float d = min(max(-sdSphere(p, 7.5), sdBox(p, vec3(6.0))), -sdSphere(p, 25.0));
	return d < 0.0;
}

float opUnion( float d1, float d2 ) {  return min(d1,d2); }
float opSubtraction( float d1, float d2 ) { return max(-d1,d2); }
 
float opRep(in vec3 p, in vec3 c)
{
    vec3 q = mod(p+0.5*c,c)-0.5*c;
    return sdSphere(q, 0.05);
}


float opRepLim( in vec3 p, in float c, in vec3 l)
{
    vec3 q = p-c*clamp(round(p/c),-l,l); // origin of sphere
    return sdSphere(q, 0.05); // probe radius here
}

float sceneSDF(vec3 point, ivec3 probeCount, float sideLength) {

	return opRepLim(point, 1.0, vec3(probeCount / 2)); // how many in each direction (right now it's 20 * 20 * 20)
}

vec3 estimateNormal(vec3 pos) {

    float epsilon = 0.0001;
    vec3 normal = vec3(0);

    normal.x = sceneSDF(vec3(pos.x + epsilon, pos.y, pos.z))
              - sceneSDF(vec3(pos.x - epsilon, pos.y, pos.z));
    normal.y = sceneSDF(vec3(pos.x, pos.y + epsilon, pos.z))
              - sceneSDF(vec3(pos.x, pos.y - epsilon, pos.z));
    normal.z = sceneSDF(vec3(pos.x, pos.y, pos.z + epsilon))
              - sceneSDF(vec3(pos.x, pos.y, pos.z - epsilon));

    return normalize(normal);
}


// this is what ray traces the probes
bool implicit_surface(Ray ray, float mint, float maxt, ivec3 probeCount, float sideLength, out Isect info) {

// start of signed distance
	vec3 ray_origin = ray.origin;
	vec3 curr_cell = vec3(floor(ray.origin));
	vec3 ray_dir = normalize(ray.direction); 

	float curr_t = 0.f;
	bool isec = false;
	// signed distance ray marching
	while (curr_t < (100)) {

		// current point along the ray
		vec3 point = ray_origin + curr_t * ray_dir;
		float dist = sceneSDF(point);

		if (dist < 0.001) {
			info.t = curr_t;
            info.normal = estimateNormal(point);
           	return true;
		}
		curr_t += dist;
	}
	return false;
//end of signed distance

}

// marches along ray and checks for blocks at locations
// changed from code given in CIS460
bool grid_march(Ray ray, float mint, float maxt, out Isect info) {
	vec3 ray_origin = ray.origin;
	vec3 curr_cell = vec3(floor(ray.origin));
	vec3 ray_dir = normalize(ray.direction);

	vec3 t2;
	float curr_t = 0.0;
	for (int i = 0; i < 200; i++) {
	    // calculate distance to voxel boundary
        t2 = max((-fract(ray_origin))/ray_dir, (1.-fract(ray_origin))/ray_dir);
        // go to next voxel
        curr_t += (min(min(t2.x, t2.y), t2.z)+.00001);
        ray_origin = ray.origin + ray_dir * curr_t;
        // get voxel's center
        vec3 pi=ceil(ray_origin) - 0.5;

        if (getBlockAt(ceil(ray_origin))) {
        	info.t = curr_t;

        	vec3 diff = normalize(ray_origin - pi);

        	vec3 normal = vec3(0, 0, 0);
        	float max = 0.0;
        	for (int i = 0; i < 3; i++) {
        		if (abs(diff[i]) > max) {
        			max = diff[i];
        			normal = vec3(0);
        			normal[i] = sign(diff[i]) * 1; 
        		}
        	}

        	info.normal = diff;
        	return true;
        }
    }

// start of signed distance 
/*
	float curr_t = 0.f;
	bool isec = false;
	while (curr_t < (100)) {

		vec3 point = ray_origin + curr_t * ray_dir;
		float dist = sceneSDF(point);

		if (dist < 0.001) {
			info.t = curr_t;
			//info.normal = vec3(1, 0, 0);
            info.normal = estimateNormal(point);
            //info.pos = point + 0.01 * info.normal;
           	return true;
		} 

		curr_t += dist;
*/
//end of signed distance


/*
		// find distance to closest axis
		float min_t = 1.0;

		//sdBox(vec3(0.5, 0.5, 0.5), vec3(1, 1, 1));
		
		for (int i = 0; i < 3; i++) {
			if (ray_dir[i] != 0) {
				float direction = sign(ray_dir[i]);
				vec3 current_pos = ray_origin + curr_t * ray_dir;
				// if we are going in the negative direction
				if (direction < 0.f) {
					// if we are going in the negative direction, we want to find the
					// closest FLOORED axis
					float dist_to_axis = abs(current_pos[i] - floor(current_pos[i]));
					if (dist_to_axis > 0.0001 && dist_to_axis < min_t) {
						min_t = dist_to_axis + 0.00001;
					}

				}
				// if we are going in the positive direction
				else if (direction > 0.f) {
					// if we are going in the negative direction, we want to find the
					// closest ceil axis
					float dist_to_axis = abs(ceil(current_pos[i]) - current_pos[i]);
					if (dist_to_axis > 0.0001 && dist_to_axis < min_t) {
						min_t = dist_to_axis + 0.00001;
					}
				}
			}
		}

		/*
		int interface_axis = -1; // Track axis for which t is smallest
		
		for(int i = 0; i < 3; ++i) { // Iterate over the three axes 
			if(ray_dir[i] != 0) { // Is ray parallel to axis i?
				float offset = max(0.f, sign(ray_dir[i]));
				// If the player is *exactly* on an interface then
            	// they'll never move if they're looking in a negative direction
            	if(curr_cell[i] == ray_origin[i] && offset == 0.f) {
            		offset = -1.f;
            	}

            	float next_inter = curr_cell[i] + offset;
            	float axis_t = (next_inter - ray_origin[i]) / ray_dir[i];

            	
            	//axis_t = min(axis_t, maxt); // Clamp to max len to avoid super out of bounds errors
            	
            	if (axis_t < min_t && axis_t > 0.f) {
            		min_t = axis_t;
            	    interface_axis = i;
            	}
            }
		}
		*/

		/*
		if(interface_axis == -1) {
        	return false;
        }  */

/*        curr_t += min_t;

        
        //ray_origin += ray_dir * min_t;
        //vec3 offset = vec3(0, 0, 0);

        // Sets it to 0 if sign is +, -1 if sign is -
        //offset[interface_axis] = min(0.f, sign(ray_dir[interface_axis]));
        curr_cell = vec3(floor(ray_origin + curr_t * ray_dir));

        // If currCell contains something other than EMPTY, return
        // curr_t
        bool hit_block = getBlockAt(curr_cell);
        
        if (hit_block == true) {
        //if (curr_cell[0] == 0.f) {
        	/*
        	Isect temp_isect;
        	temp_isect.t = curr_t;
        	temp_isect.pos = curr_t * ray.direction + ray.origin; 

        	// calculate normal 
        	//vec3 curr_cell_center = 
        	temp_isect.normal = vec3(1, 0, 0);

        	
			//Material mat;
			//mat.albedo = vec4(1, 0, 0, 0);
    		//mat.emission = vec4(1, 1, 1, 0);
    		//mat.data = vec4(0, 0, 0, 0);

    		//temp_isect.mat = convert_old_material(mat);

    		info = temp_isect;

            return true;
            */
            

            //info.t = curr_t;
            //info.pos = curr_t * ray_dir + ray.origin;
           // info.normal = vec3(1, 0, 0);
           // return true;
            //break;
        //}

	//}

	return false;
}

/*
bool grid_march2(Ray ray, float mint, float maxt, out Isect info) {
	vec3 rayOrigin = ray.origin;
	float maxLen = length(ray.direction); // Farthest we search
    vec3 currCell = vec3(floor(rayOrigin));
    vec3 rayDirection = normalize(ray.direction); // Now all t values represent world dist.

    float curr_t = 0.f;
    while(curr_t < maxLen) {
        float min_t = sqrt(3.f);
        float interfaceAxis = -1; // Track axis for which t is smallest
        for(int i = 0; i < 3; ++i) { // Iterate over the three axes
            if(rayDirection[i] != 0) { // Is ray parallel to axis i?
                float offset = max(0.f, sign(rayDirection[i])); // See slide 5
                // If the player is *exactly* on an interface then
                // they'll never move if they're looking in a negative direction
                if(currCell[i] == rayOrigin[i] && offset == 0.f) {
                    offset = -1.f;
                }
                int nextIntercept = currCell[i] + offset;
                float axis_t = (nextIntercept - rayOrigin[i]) / rayDirection[i];
                axis_t = min(axis_t, maxLen); // Clamp to max len to avoid super out of bounds errors
                if(axis_t < min_t) {
                    min_t = axis_t;
                    interfaceAxis = i;
                }
            }
        }
        if(interfaceAxis == -1) {
        	return false;
            //throw std::out_of_range("interfaceAxis was -1 after the for loop in gridMarch!");
        }
        curr_t += min_t; // min_t is declared in slide 7 algorithm
        rayOrigin += rayDirection * min_t;
        vec3 offset = vec3(0,0,0);
        // Sets it to 0 if sign is +, -1 if sign is -
        offset[interfaceAxis] = min(0.f, sign(rayDirection[interfaceAxis]));
        currCell = vec3(floor(rayOrigin)) + offset;
        // If currCell contains something other than EMPTY, return
        // curr_t
        BlockType cellType = terrain.getBlockAt(currCell.x, currCell.y, currCell.z);
        if(cellType != EMPTY) {
            *out_blockHit = currCell;
            *out_dist = glm::min(maxLen, curr_t);
            return true;
        }
    }
    *out_dist = min(maxLen, curr_t);
    return false;
} */

bool intersect_probes (
	 Ray  ray,  /* ray for the intersection */
	 float     mint, /* lower bound for t */
	 float     maxt, /* upper bound for t */
	 ivec3 probeCount,
	 float sideLength,
	 out Isect info) /* intersection data */ 
{
	float closest_t = INF;
	info.t = closest_t;
	info.pos = vec3(0);
	info.normal = vec3(0);
	Isect temp_isect;

	if (implicit_surface(ray, mint, maxt, probeCount, sideLength, temp_isect)) {
		info = temp_isect;
		closest_t = info.t;

		info.normal = closest_t<INF? normalize(info.normal) : vec3(0);
					
		info.pos = closest_t<INF? ray.origin + info.t * ray.direction : vec3(0);

		info.pos += 0.001 * info.normal;
	
		return true;
	}
	return false;
}


/*--------------------------------------------------------------------------*/

bool intersect_scene

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

	/* intersect triangles */
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

	// CHANGED: added floor
	intersect_plane(ray, -0.5, vec3(0, 1, 0), mint, maxt, temp_isect);
	if (temp_isect.t<closest_t)
	{
		info = temp_isect;
		Material mat = materials[2]; // TO DO: Don't hard code this
		info.mat = convert_old_material(mat);
	}
	closest_t = min(temp_isect.t, closest_t);
	
	info.normal = closest_t<INF? normalize(info.normal) : vec3(0);
					
	info.pos = closest_t<INF? ray.origin + info.t * ray.direction : vec3(0);
	
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