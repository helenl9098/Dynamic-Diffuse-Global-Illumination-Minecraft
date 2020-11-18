//
// Created by AregevDev on 23/04/2020.
//

#include <imgui.h>
#include <fmt/core.h>
#include "rvpt.h"

#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
#include "tinyobjloader/tiny_obj_loader.h"

void load_model(RVPT& rvpt, std::string inputfile, int material_id)
{
    rvpt.get_asset_path(inputfile);

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    std::string warn;
    std::string err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, inputfile.c_str());

    if (!warn.empty())
    {
        fmt::print("[{}: {}] {}\n", "WARNING", "MODEL-LOADING", warn);
    }

    if (!err.empty())
    {
        fmt::print("[{}: {}] {}\n", "ERROR", "MODEL-LOADING", err);
        exit(-1);
    }

    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            int fv = shapes[s].mesh.num_face_vertices[f];

            // Loop over vertices in the face.
            if (fv != 3)
            {
                fmt::print("Shape had a face with more than 3 vertices, skipping");
                continue;
            }
            glm::vec3 vertices[3];

            for (size_t v = 0; v < fv; v++) {
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                vertices[v].x = attrib.vertices[3*idx.vertex_index+0];
                vertices[v].y = attrib.vertices[3*idx.vertex_index+1];
                vertices[v].z = attrib.vertices[3*idx.vertex_index+2];
            }
            index_offset += fv;

            rvpt.add_triangle(Triangle(vertices[0], vertices[1], vertices[2], material_id));

            // per-face material
            shapes[s].mesh.material_ids[f];
        }
    }

}

void update_camera(Window& window, RVPT& rvpt)
{
    glm::vec3 movement{};
    double frameDelta = rvpt.time.since_last_frame();

    if (window.is_key_held(Window::KeyCode::KEY_LEFT_CONTROL)) frameDelta *= 5;
    if (window.is_key_held(Window::KeyCode::SPACE)) movement.y += 3.0f;
    if (window.is_key_held(Window::KeyCode::KEY_LEFT_SHIFT)) movement.y -= 3.0f;
    if (window.is_key_held(Window::KeyCode::KEY_W)) movement.z += 3.0f;
    if (window.is_key_held(Window::KeyCode::KEY_S)) movement.z -= 3.0f;
    if (window.is_key_held(Window::KeyCode::KEY_D)) movement.x += 3.0f;
    if (window.is_key_held(Window::KeyCode::KEY_A)) movement.x -= 3.0f;

    rvpt.scene_camera.move(static_cast<float>(frameDelta) * movement);

    glm::vec3 rotation{};
    float rot_speed = 0.3f;
    if (window.is_key_down(Window::KeyCode::KEY_RIGHT)) rotation.x = rot_speed;
    if (window.is_key_down(Window::KeyCode::KEY_LEFT)) rotation.x = -rot_speed;
    if (window.is_key_down(Window::KeyCode::KEY_UP)) rotation.y = -rot_speed;
    if (window.is_key_down(Window::KeyCode::KEY_DOWN)) rotation.y = rot_speed;
    rvpt.scene_camera.rotate(rotation);
}

int main()
{
    Window::Settings settings;
    settings.width = 2000;
    settings.height = 2000;
    Window window(settings);

    RVPT rvpt(window);

    load_model(rvpt, "models/cube.obj", 1);

    // Setup Demo Scene
    rvpt.add_material(Material(glm::vec4(0, 0, 1, 0), glm::vec4(2.0, 2.0, 2.0, 0),
                               Material::Type::LAMBERT));
    rvpt.add_sphere(Sphere(glm::vec3(1, 1.5, 1.5), 0.4f, 0));
    rvpt.add_material(Material(glm::vec4(1.0, 0.0, 0.0, 0), glm::vec4(0), Material::Type::LAMBERT));
    rvpt.add_material(Material(glm::vec4(0.0, 1.0, 0.0, 0), glm::vec4(0), Material::Type::LAMBERT));

    bool rvpt_init_ret = rvpt.initialize();
    if (!rvpt_init_ret)
    {
        fmt::print("failed to initialize RVPT\n");
        return 0;
    }

    window.setup_imgui();
    window.add_mouse_move_callback([&window, &rvpt](double x, double y) {
        if (window.is_mouse_locked_to_window())
        {
            rvpt.scene_camera.rotate(glm::vec3(x * 0.3f, -y * 0.3f, 0));
        }
    });

    window.add_mouse_click_callback([&window](Window::Mouse button, Window::Action action) {
        if (button == Window::Mouse::LEFT && action == Window::Action::RELEASE &&
            window.is_mouse_locked_to_window())
        {
            window.set_mouse_window_lock(false);
        }
        else if (button == Window::Mouse::LEFT && action == Window::Action::RELEASE &&
                 !window.is_mouse_locked_to_window())
        {
            if (!ImGui::GetIO().WantCaptureMouse)
            {
                window.set_mouse_window_lock(true);
            }
        }
    });
    while (!window.should_close())
    {
        window.poll_events();
        if (window.is_key_down(Window::KeyCode::KEY_ESCAPE)) window.set_close();
        if (window.is_key_down(Window::KeyCode::KEY_R)) rvpt.reload_shaders();
        if (window.is_key_down(Window::KeyCode::KEY_V)) rvpt.toggle_debug();
        if (window.is_key_up(Window::KeyCode::KEY_ENTER))
        {
            window.set_mouse_window_lock(!window.is_mouse_locked_to_window());
        }

        update_camera(window, rvpt);
        ImGui::NewFrame();
        rvpt.update_imgui();
        rvpt.update();
        rvpt.draw();
    }
    rvpt.shutdown();

    return 0;
}
