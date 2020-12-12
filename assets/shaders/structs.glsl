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

const int num_lights_0 = 2;
const int num_lights_1 = 2;
const int num_lights_2 = 2;

int num_lights[3] = {2, 2, 2};

Light lights_0[num_lights_0] = {{1.f, vec3(1.f), vec3(4, 17.5, 8.5)},
                                {1.f, vec3(1.f), vec3(4, 17.5, 8.5)}};

Light lights_1[num_lights_1] = {{1.f, vec3(1.f), vec3(0, 8, 13)}, {1.f, vec3(1.f), vec3(0, 8, 13)}};

Light lights_2[num_lights_2] = {{1.f, vec3(1.f), vec3(5, 9.3, 36.5)},
                                {1.f, vec3(1.f), vec3(5, 9.3, 36.5)}};