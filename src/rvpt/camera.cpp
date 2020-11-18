//
// Created by legend on 5/26/20.
//
#include "camera.h"

#define GLM_DEPTH_ZERO_TO_ONE
#include <glm/ext.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <imgui.h>

#include "imgui_helpers.h"

constexpr glm::vec3 RIGHT = glm::vec3(1, 0, 0);
constexpr glm::vec3 UP = glm::vec3(0, 1, 0);
constexpr glm::vec3 FORWARD = glm::vec3(0, 0, 1);

glm::mat4 construct_camera_matrix(glm::vec3 translation, glm::vec3 rotation)
{
    auto mat = glm::mat4{1.f};
    mat = glm::translate(mat, translation);
    mat = glm::rotate(mat, glm::radians(rotation.x), UP);
    mat = glm::rotate(mat, glm::radians(rotation.y), RIGHT);
    mat = glm::rotate(mat, glm::radians(rotation.z), FORWARD);
    return mat;
}

Camera::Camera(float aspect, glm::vec3 origin, glm::vec3 rotation) : aspect(aspect), translation(origin), rotation(rotation) { 
    
    // TO DO: Don't manually set FOV 
    fov = 45;

    recalculate_values(); 

}

void Camera::move(glm::vec3 translation)
{
    auto translation_mat = construct_camera_matrix(this->translation, this->rotation);
    this->translation += glm::vec3(translation_mat * glm::vec4(translation, 0));
}

void Camera::rotate(glm::vec3 rotation)
{
    this->rotation += rotation;
    if (vertical_view_angle_clamp) this->rotation.y = glm::clamp(this->rotation.y, -90.f, 90.f);
}
void Camera::set_fov(float in_fov) { fov = in_fov; }

float Camera::get_fov() { return fov; }

void Camera::set_scale(float in_scale) { scale = in_scale; }

float Camera::get_scale() { return scale; }

void Camera::set_camera_mode(int in_mode) { mode = in_mode; }

int Camera::get_camera_mode() { return mode; }

void Camera::clamp_vertical_view_angle(bool clamp) { vertical_view_angle_clamp = clamp; }

void Camera::recalculate_values()
{
    float theta = glm::radians(fov);
    float view_height = tanf(theta / 2);
    float view_width = aspect * view_height;

    camera_matrix = construct_camera_matrix(this->translation, this->rotation);

    view_matrix = glm::inverse(camera_matrix);

    if (mode == 0)
    {
        pv_matrix = glm::perspectiveLH_ZO(glm::radians(fov), aspect, 0.1f, 1000.f) * view_matrix;
    }
    else if (mode == 1)
    {
        pv_matrix = glm::orthoLH_ZO(-view_width * scale, view_width * scale, -view_height * scale,
                                    view_height * scale, 0.1f, 1000.f) *
                    view_matrix;
    }
}

std::vector<glm::vec4> Camera::get_data()
{
    recalculate_values();
    std::vector<glm::vec4> data;
    data.emplace_back(camera_matrix[0]);
    data.emplace_back(camera_matrix[1]);
    data.emplace_back(camera_matrix[2]);
    data.emplace_back(camera_matrix[3]);
    data.emplace_back(aspect, glm::radians(fov), scale, 0);

    return data;
}

glm::mat4 Camera::get_camera_matrix()
{
    recalculate_values();
    return camera_matrix;
}

glm::mat4 Camera::get_view_matrix()
{
    recalculate_values();
    return view_matrix;
}

glm::mat4 Camera::get_pv_matrix()
{
    recalculate_values();
    return pv_matrix;
}

void Camera::update_imgui()
{
    static bool is_active = true;
    ImGui::SetNextWindowPos({0, 265}, ImGuiCond_Once);
    ImGui::SetNextWindowSize({200, 210}, ImGuiCond_Once);

    if (ImGui::Begin("Camera Data", &is_active))
    {
        ImGui::PushItemWidth(125);
        ImGui::DragFloat3("position", glm::value_ptr(translation), 0.2f);
        ImGui::DragFloat3("rotation", glm::value_ptr(rotation), 0.2f);
        ImGui::Text("Reset");
        ImGui::SameLine();
        if (ImGui::Button("Pos")) translation = {};

        ImGui::SameLine();
        if (ImGui::Button("Rot")) rotation = {};

        ImGui::Text("Projection");
        dropdown_helper("camera_mode", mode, CameraModes);
        if (mode == 0)
        {
            ImGui::SliderFloat("fov", &fov, 1, 179);
        }
        else if (mode == 1)
        {
            ImGui::SliderFloat("scale", &scale, 0.1, 20);
        }

        ImGui::Checkbox("Clamp Vertical Rot", &vertical_view_angle_clamp);

        static bool show_view_matrix = false;
        ImGui::Checkbox("Show View Matrix", &show_view_matrix);
        if (show_view_matrix)
        {
            ImGui::PushItemWidth(170);
            ImGui::DragFloat4("", glm::value_ptr(camera_matrix[0]), 0.05f);
            ImGui::DragFloat4("", glm::value_ptr(camera_matrix[1]), 0.05f);
            ImGui::DragFloat4("", glm::value_ptr(camera_matrix[2]), 0.05f);
            ImGui::DragFloat4("", glm::value_ptr(camera_matrix[3]), 0.05f);
        }
    }
    ImGui::End();
    recalculate_values();
}
