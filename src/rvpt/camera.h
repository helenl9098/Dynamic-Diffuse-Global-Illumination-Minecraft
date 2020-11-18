//
// Created by legend on 5/26/20.
//

#pragma once

#include <vector>
#include <glm/glm.hpp>

static const char* CameraModes[] = {"perspective", "orthographic", "spherical"};

class Camera
{
public:
    explicit Camera(float aspect, glm::vec3 origin, glm::vec3 rotation);

    void move(glm::vec3 translation);
    void rotate(glm::vec3 rotation);

    void set_fov(float fov);
    float get_fov();

    void set_scale(float scale);
    float get_scale();

    void set_camera_mode(int mode);
    int get_camera_mode();

    // Prevent looking beyond +/- 90 degrees vertically
    void clamp_vertical_view_angle(bool clamp);

    void position_lock(bool locked);
    void rotation_lock(bool locked);

    void update_imgui();

    std::vector<glm::vec4> get_data();
    glm::mat4 get_camera_matrix();
    glm::mat4 get_view_matrix();
    glm::mat4 get_pv_matrix();

private:
    void recalculate_values();
    int mode = 0;
    float fov = 90.f, scale = 4.f, aspect{};
    bool vertical_view_angle_clamp = false;

    glm::vec3 translation{};
    glm::vec3 rotation{};

    glm::mat4 camera_matrix{1.f};

    /* view = inverse(camera_matrix) */
    glm::mat4 view_matrix{1.f};
    /* projection * view matrix */
    glm::mat4 pv_matrix{1.f};
};
