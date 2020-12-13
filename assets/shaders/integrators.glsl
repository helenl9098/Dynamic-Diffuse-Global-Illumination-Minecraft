/*--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------*/
/*                                                                          */
/*                                                                          */
/*                             INTEGRATORS					                */
/*                   									                    */
/*                                                                          */
/*--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------*/

/*
        TODO:

        - Add rasterizer-like integrator.
        - Add specific bounce integrator.
        - Add Cook integrator?
    - Add heat map LUT for Hart.
*/

const float TWO_PI = 6.28318531;
const float SQRT_OF_ONE_THIRD = 0.577350269;

/*--------------------------------------------------------------------------*/

vec3 integrator_DDGI

	(Ray   ray,          /* primary ray */
	 float mint,         /* lower bound for t */
	 float maxt,	     /* upper bound for t */
	 ivec3 probe_counts, /* the dimensions of the irradiance field */
	 int side_length,    /* the distance between two probes in the field */
     vec3 field_origin)  /* the world space position of the center of the field */
	 
/*
	Returns global illumination, i.e. direct lighting plus 
    indirect lighting that is sampled from the probes
*/
{
    Isect info;
    bool intersect = intersect_scene(ray, mint, maxt, info);
    Isect temp_info;
    
    if(render_settings.visualize_probes) {    // Probes visualization here
        vec3 probe_pos = vec3(0);
        vec3 tgt_pos = vec3(2, 2, 2);
        ivec3 tgt_box = ivec3(floor(tgt_pos / float(side_length)));
        if (intersect_probes(ray, mint, maxt, probe_counts, side_length, temp_info, probe_pos,
                             field_origin))
        {
            if (temp_info.t < info.t)
            {
                // uncomment if you want there to be a check probes around a certain point
                /*for (int i = 0; i < 8; i++) {
                    ivec3 offset = ivec3(i >> 2, i >> 1, i) & ivec3(1);
                    ivec3 test_probe_pos = ivec3(round((tgt_box + offset) * side_length));
                    if (distance(probe_pos, test_probe_pos) <= 0.001) {
                        return vec3(1, 0, 1);
                        break;
                    }
                } */

                return vec3(0, 1, 1);  // probe color here
            }
        }
    }

    vec3 col = vec3(0.90, 0.90, 1.0);
    if (!intersect) return col;

    // returns the light color if a light sphere is hit
    if (info.type == 2) return info.mat.emissive;

    vec3 indirect_lighting = get_diffuse_gi(info, probe_counts, side_length, ray);

    // this is just a hack so the light feeler ray can be calculated by the get intersection
    vec3 direct_lighting = vec3(0.f);
    int num_visible_lights = 0;
    for (int i = 0; i < num_lights[render_settings.scene]; i++)
    {
        Light l = get_light(render_settings.scene, i);
        Ray light_feeler = Ray(info.pos, normalize(l.pos - info.pos));
        if (intersect_scene(light_feeler, mint, maxt, temp_info))
        {
            if (temp_info.type == 2)
            {
                float lambert =
                    clamp(dot(normalize(info.normal),
                              normalize(l.pos - info.pos)),
                          0.0, 1.0);
                float dist = distance(l.pos, info.pos);
                direct_lighting += lambert * l.col * l.intensity / (dist);
                num_visible_lights++;
            }
        }
    }

    if (num_visible_lights != 0)
    {
        return 0.5 * info.mat.base_color * (direct_lighting / float(num_visible_lights)) +
               0.5 * info.mat.base_color * indirect_lighting;
    }
    return 0.5 * indirect_lighting * info.mat.base_color;

} /* integrator_DDGI */

/*--------------------------------------------------------------------------*/

vec3 integrator_direct

    (Ray   ray,          /* primary ray */
     float mint,         /* lower bound for t */
     float maxt         /* upper bound for t */
    )

 /*
    Returns only the direct lighting.
*/
     
{
    Isect info;
    bool intersect = intersect_scene(ray, mint, maxt, info);
    Isect temp_info;
    
    vec3 col = vec3(0.0, 0.0, 0.0);
    if (!intersect)
        return col;

    vec3 direct_lighting = vec3(0.f);
    int num_visible_lights = 0;
    for (int i = 0; i < num_lights[render_settings.scene]; i++)
    {
        Light l = get_light(render_settings.scene, i);
        Ray light_feeler = Ray(info.pos, normalize(l.pos - info.pos));
        if (intersect_scene(light_feeler, mint, maxt, temp_info))
        {
            if (temp_info.type == 2)
            {
                float lambert =
                    clamp(dot(normalize(info.normal), normalize(l.pos - info.pos)), 0.0, 1.0);
                float dist = distance(l.pos, info.pos);
                direct_lighting += lambert * l.col * l.intensity / (dist);
                num_visible_lights++;
            }
        }
    }

    if (num_visible_lights != 0)
    {
        return 0.5 * info.mat.base_color * (direct_lighting / float(num_visible_lights));
    }

    return col;

} /* integrator_direct */

/*--------------------------------------------------------------------------*/

vec3 integrator_indirect

    (Ray   ray,     /* primary ray */
     float mint,    /* lower bound for t */
     float maxt,    /* upper bound for t */
     ivec3 probe_counts, /* the dimensions of the irradiance field */
     int side_length,    /* the distance between two probes in the field */
     vec3 field_origin)  /* the world space position of the center of the field */
     
/*
    Returns only the indirect lighting that is sampled from probes,
*/
     
{

    Isect info;
    bool intersect = intersect_scene(ray, mint, maxt, info);
    Isect temp_info;

    
    if(render_settings.visualize_probes == true) { // Probes visualization here
        vec3 probe_pos = vec3(0);
        vec3 tgt_pos = vec3(2, 2, 2);
        ivec3 tgt_box = ivec3(floor(tgt_pos / float(side_length)));
        if (intersect_probes(ray, mint, maxt, probe_counts, side_length, temp_info, probe_pos, field_origin)) {
            if (temp_info.t < info.t) {
                // uncomment if you want there to be a check probes around a certain point
                /*for (int i = 0; i < 8; i++) {
                    ivec3 offset = ivec3(i >> 2, i >> 1, i) & ivec3(1);
                    ivec3 test_probe_pos = ivec3(round((tgt_box + offset) * side_length));
                    if (distance(probe_pos, test_probe_pos) <= 0.001) {
                        return vec3(1, 0, 1);
                        break;
                    }
                }*/

                return vec3(0, 1, 1); // probe color here
            }
        } 
    }
    
    vec3 col = vec3(0, 0, 0);
    if (!intersect)
        return col;

    return 0.5 * get_diffuse_gi(info, probe_counts, side_length, ray);

} /* integrator_indirect */

/*--------------------------------------------------------------------------*/

vec3 integrator_color

    (Ray ray,    /* primary ray */
     float mint, /* lower bound for t */
     float maxt) /* upper bound for t */

/*
        Returns the base color of the intersected surface, otherwise (0,0,0).
*/

{
    Isect info;
    if (!intersect_scene(ray, mint, maxt, info))
        return vec3(0);
    else
        return info.mat.base_color;

} /* integrator_color */

/*--------------------------------------------------------------------------*/

vec3 integrator_depth

    (Ray ray,    /* primary ray */
     float mint, /* lower bound for t */
     float maxt) /* upper bound for t */

/*
        Returns the reciprocal distance to the intersection, measured
        from the primary ray's origin.
*/

{
    Isect info;
    intersect_scene(ray, mint, maxt, info);

    /* find the distance by taking into account direction's magnitude */
    float inv_dist = 1.0 / (length(ray.direction) * info.t);
    return vec3(inv_dist);

} /* integrator_depth */

/*--------------------------------------------------------------------------*/

vec3 integrator_normal

    (Ray ray,    /* primary ray */
     float mint, /* lower bound for t */
     float maxt) /* upper bound for t */

/*
        Upon intersection returns a color constructed as:
        0.5 * normal + 0.5. If there is no intersection returns (0,0,0).
*/

{
    Isect info;
    float isect = float(intersect_scene(ray, mint, maxt, info));
    return 0.5 * info.normal + vec3(0.5 * isect);

} /* integrator_normal */
