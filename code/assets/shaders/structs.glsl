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