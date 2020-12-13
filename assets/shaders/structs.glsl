struct Sphere
{
    vec3 origin;
    float radius;
    vec4 mat_id;
};

struct Triangle
{
    vec4 vert0;
    vec4 vert1;
    vec4 vert2;
    vec4 mat_id;
};

struct Ray
{
    vec3 origin;
    vec3 direction;
};

struct ProbeRay
{
    vec3 origin;
    vec3 direction;
    vec3 probe_info;
};

struct Material
{
    vec4 albedo;
    vec4 emission;
    vec4 data;
    /*
    data.x = Type (Glass, Lambert, Dynamic)
    data.y = Glass Refractive Index OR Dynamic difusse
    data.z = Dynamic reflectiveness
    data.w = Unused
    */
};

struct Record
{
    bool hit;
    float distance;
    float reflectiveness;
    vec3 emission;
    vec3 albedo;
    vec3 normal;
    vec3 intersection;
    Material mat;
};

struct Light
{
    float intensity;
    vec3 col;
    vec3 pos;
};

const int num_lights[3] = {1, 3, 2};


 Light lights_0[num_lights[0]] = {{10.f, vec3(1.f), vec3(4, 17.5, 8.5)}};

/*Light lights_0[num_lights[0]] = {{20.f, vec3(1.f), vec3(4, 17.5, 8.5)},
                                 {10.f, vec3(1.f, 0.1f, 0.1f), vec3(0, 0, 0)},
                                 {10.f, vec3(0.1f, 0.1f, 1.f), vec3(5, 0, 0)},
                                 {10.f, vec3(0.1f, 1.f, 0.1f), vec3(0, 5, 0)},
};*/



/* Light lights_0[num_lights[0]] = {{10.f, vec3(1.f), vec3(4, 17.5, 8.5)},
                                 {10.f, vec3(1.f, 0.1f, 0.1f), vec3(-19, 14, 13)},
                                 {10.f, vec3(0.1f, 0.1f, 1.f), vec3(-17, 8, 9)},
                                 {10.f, vec3(0.1f, 1.f, 0.1f), vec3(-3, 14, 9)},
                                 {10.f, vec3(0.1f, 1.f, 0.1f), vec3(-8, 11, 8.5)},
                                 {10.f, vec3(0.1f, 1.f, 0.1f), vec3(4, 11, 8.5)},
                                 {10.f, vec3(0.1f, 1.f, 0.1f), vec3(0, 11, 8.5)},
                                 {10.f, vec3(0.1f, 1.f, 0.1f), vec3(-11, 11, 8.5)}
}; */

//Light lights_1[num_lights[1]] = {{1.f, vec3(1.f), vec3(0, 8, 13)},
//                                 {1.f, vec3(1.f), vec3(0, 8, 9)}};

Light lights_1[num_lights[1]] = {{20.f, vec3(1.f, 0.1f, 0.1f), vec3(0, 8, 13)},
                                {20.f, vec3(0.1f, 0.1f, 1.f), vec3(0, 8, 9)},
                                {20.f, vec3(0.1f, 1.f, 0.1f), vec3(-3, 8, 9)},
                                };

Light lights_2[num_lights[2]] = {{1.f, vec3(1.f), vec3(5, 9.3, 36.5)},
                                {1.f, vec3(1.f), vec3(0, 0, 0)}};