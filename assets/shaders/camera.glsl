/*--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------*/
/*                                                                          */
/*                                                                          */
/*                                CAMERA					                */
/*                   									                    */
/*                                                                          */
/*--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------*/

/*
	TODO:
	
	- Fix fov change on cpp side to update the camera: it keeps accumulating 
	frames otherwise.
	- Add sphere cap camera (parametrized by angle).
	- Add thin-lens camera.
	- Add system of lenses camera (Kolb)?
	- Fix ortho to accept scale parameters.
	- Panini projection?
	- Arbitrary mesh camera with bijective texture UVs?
	- Add image/arbitrary shape apertures?
*/

/*--------------------------------------------------------------------------*/

Ray camera_pinhole_ray

	(float x,    /* x coordinate in [0, 1] */
	 float y)    /* y coordinate in [0, 1] */
	 
/*
	Constructs a ray passing through the film point with 
	(local) coordinates (x,y) for a pinhole/perspective camera.
*/
 
{
	float aspect = cam.params.x;
	float hfov = cam.params.y;
	float u = aspect * (2.0*x-1.0);
	float v = 2.0*y-1.0;
	float w = 1.0/tan(0.5*hfov);

    vec3 origin = cam.matrix[3].xyz;
	
    vec3 direction = (cam.matrix * vec4(u,v,w,0.0)).xyz;
    return Ray(origin, normalize(direction));

} /* camera_pinhole_ray */

/*--------------------------------------------------------------------------*/

Ray camera_ortho_ray

	(float x,    /* x coordinate in [0, 1] */
	 float y)    /* y coordinate in [0, 1] */
	 
/*
	Constructs a ray passing through the film point with 
	(local) coordinates (x,y) for an orthographic camera.
*/
 
{
	float scale_x = cam.params.z;
	float scale_y = cam.params.z;
	float aspect = cam.params.x;
	float u = aspect * (2.0*x-1.0);
	float v = 2.0*y-1.0;
	
    vec3 origin = (cam.matrix * vec4(scale_x*u,scale_y*v,0.0,1.0)).xyz;
    vec3 direction = cam.matrix[2].xyz;
    return Ray(origin, direction);

} /* camera_ortho_ray */

/*--------------------------------------------------------------------------*/

Ray camera_spherical_ray

	(float x,    /* x coordinate in [0, 1] */
	 float y)    /* y coordinate in [0, 1] */
	 
/*
	Constructs a ray passing through the film point with 
	(local) coordinates (x,y) for a spherical/environment camera.
*/
 
{
	float phi = x * 2 * PI;
	float theta = y * PI;
	
	vec3 origin = cam.matrix[3].xyz;
	vec3 local_dir = unit_spherical_to_cartesian(phi, theta).xzy;
    vec3 direction = (cam.matrix * vec4(local_dir, 0)).xyz;
    return Ray(origin, direction);

} /* camera_spherical_ray */

/*--------------------------------------------------------------------------*/