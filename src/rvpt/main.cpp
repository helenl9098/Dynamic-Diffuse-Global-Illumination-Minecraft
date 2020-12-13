//
// Created by AregevDev on 23/04/2020.
//

#include <imgui.h>
#include <fmt/core.h>
#include "rvpt.h"


void update_camera(Window& window, RVPT& rvpt)
{
    glm::vec3 movement{};
    double frameDelta = rvpt.time.since_last_frame();
    // float mvmt_speed = 3.f;
    float mvmt_speed = 10.f;

    if (window.is_key_held(Window::KeyCode::KEY_LEFT_CONTROL)) frameDelta *= 5;
    if (window.is_key_held(Window::KeyCode::SPACE)) movement.y += mvmt_speed;
    if (window.is_key_held(Window::KeyCode::KEY_LEFT_SHIFT)) movement.y -= mvmt_speed;
    if (window.is_key_held(Window::KeyCode::KEY_W)) movement.z += mvmt_speed;
    if (window.is_key_held(Window::KeyCode::KEY_S)) movement.z -= mvmt_speed;
    if (window.is_key_held(Window::KeyCode::KEY_D)) movement.x += mvmt_speed;
    if (window.is_key_held(Window::KeyCode::KEY_A)) movement.x -= mvmt_speed;

    rvpt.scene_camera.move(static_cast<float>(frameDelta) * movement);

    glm::vec3 rotation{};
    // float rot_speed = 0.3f;
    float rot_speed = 10.f;
    if (window.is_key_down(Window::KeyCode::KEY_RIGHT)) rotation.x = rot_speed;
    if (window.is_key_down(Window::KeyCode::KEY_LEFT)) rotation.x = -rot_speed;
    if (window.is_key_down(Window::KeyCode::KEY_UP)) rotation.y = -rot_speed;
    if (window.is_key_down(Window::KeyCode::KEY_DOWN)) rotation.y = rot_speed;
    rvpt.scene_camera.rotate(rotation);
}

int main()
{
    Window::Settings settings;
    settings.width = 1600;
    settings.height = 900;
    Window window(settings);

    RVPT rvpt(window);

    // Setup Demo Scene
    rvpt.generate_probe_rays();


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
        //if (window.is_key_down(Window::KeyCode::KEY_V)) rvpt.toggle_debug();
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
