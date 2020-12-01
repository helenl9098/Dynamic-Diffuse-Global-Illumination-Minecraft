//
// Created by legend on 5/20/20.
//

#include "window.h"

#include <imgui.h>
#include <fmt/core.h>

Window::Window(Window::Settings settings) : active_settings(settings)
{
    // Initializing glfw and a window
    auto glfw_ret = glfwInit();
    if (!glfw_ret) fmt::print(stderr, "Failed to initialize glfw\n");

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    window_ptr =
        glfwCreateWindow(settings.width, settings.height, settings.title, nullptr, nullptr);
    if (window_ptr == nullptr)
        fmt::print(stderr, "Failed to create a glfw window\n");
    else
    {
        glfwSetWindowUserPointer(window_ptr, this);
        glfwSetKeyCallback(window_ptr, key_callback);
        glfwSetCharCallback(window_ptr, char_callback);
        glfwSetMouseButtonCallback(window_ptr, mouse_click_callback);
        glfwSetCursorPosCallback(window_ptr, mouse_move_callback);
        glfwSetScrollCallback(window_ptr, scroll_callback);
    }

    double x = 0.0, y = 0.0;
    glfwGetCursorPos(window_ptr, &x, &y);
    mouse_position_x = mouse_position_previous_x = x;
    mouse_position_y = mouse_position_previous_y = y;
}

Window::~Window()
{
    glfwDestroyWindow(window_ptr);
    glfwTerminate();
}

void Window::setup_imgui()
{
    // Setup back-end capabilities flags
    ImGuiIO& io = ImGui::GetIO();
    io.BackendFlags |=
        ImGuiBackendFlags_HasMouseCursors;  // We can honor GetMouseCursor() values (optional)
    io.BackendFlags |= ImGuiBackendFlags_HasSetMousePos;  // We can honor io.WantSetMousePos
                                                          // requests (optional, rarely used)
    io.BackendPlatformName = "imgui_impl_glfw";

    // Keyboard mapping. ImGui will use those indices to peek into the io.KeysDown[] array.
    io.KeyMap[ImGuiKey_Tab] = GLFW_KEY_TAB;
    io.KeyMap[ImGuiKey_LeftArrow] = GLFW_KEY_LEFT;
    io.KeyMap[ImGuiKey_RightArrow] = GLFW_KEY_RIGHT;
    io.KeyMap[ImGuiKey_UpArrow] = GLFW_KEY_UP;
    io.KeyMap[ImGuiKey_DownArrow] = GLFW_KEY_DOWN;
    io.KeyMap[ImGuiKey_PageUp] = GLFW_KEY_PAGE_UP;
    io.KeyMap[ImGuiKey_PageDown] = GLFW_KEY_PAGE_DOWN;
    io.KeyMap[ImGuiKey_Home] = GLFW_KEY_HOME;
    io.KeyMap[ImGuiKey_End] = GLFW_KEY_END;
    io.KeyMap[ImGuiKey_Insert] = GLFW_KEY_INSERT;
    io.KeyMap[ImGuiKey_Delete] = GLFW_KEY_DELETE;
    io.KeyMap[ImGuiKey_Backspace] = GLFW_KEY_BACKSPACE;
    io.KeyMap[ImGuiKey_Space] = GLFW_KEY_SPACE;
    io.KeyMap[ImGuiKey_Enter] = GLFW_KEY_ENTER;
    io.KeyMap[ImGuiKey_Escape] = GLFW_KEY_ESCAPE;
    io.KeyMap[ImGuiKey_KeyPadEnter] = GLFW_KEY_KP_ENTER;
    io.KeyMap[ImGuiKey_A] = GLFW_KEY_A;
    io.KeyMap[ImGuiKey_C] = GLFW_KEY_C;
    io.KeyMap[ImGuiKey_V] = GLFW_KEY_V;
    io.KeyMap[ImGuiKey_X] = GLFW_KEY_X;
    io.KeyMap[ImGuiKey_Y] = GLFW_KEY_Y;
    io.KeyMap[ImGuiKey_Z] = GLFW_KEY_Z;

    // Setup display size (every frame to accommodate for window resizing)
    int w, h;
    int display_w, display_h;
    glfwGetWindowSize(window_ptr, &w, &h);
    glfwGetFramebufferSize(window_ptr, &display_w, &display_h);
    io.DisplaySize = ImVec2((float)w, (float)h);
    if (w > 0 && h > 0)
        io.DisplayFramebufferScale = ImVec2((float)display_w / w, (float)display_h / h);
}

float Window::get_aspect_ratio()
{
    return static_cast<float>(active_settings.width) / static_cast<float>(active_settings.height);
}

void Window::poll_events()
{
    for (auto& key : key_states)
    {
        if (key == KeyState::released) key = KeyState::none;
        if (key == KeyState::pressed) key = KeyState::held;
    }
    glfwPollEvents();
    if (mouse_locked_to_window)
    {
        glfwSetInputMode(window_ptr, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    }
}

void Window::add_mouse_click_callback(Window::MouseClickCallback callback)
{
    mouse_click_callbacks.push_back(callback);
}

void Window::add_mouse_move_callback(Window::MouseMoveCallback callback)
{
    mouse_move_callbacks.push_back(callback);
}

void Window::add_scroll_callback(Window::MouseScrollCallback callback)
{
    scroll_callbacks.push_back(callback);
}

bool Window::is_key_down(Window::KeyCode keycode)
{
    return key_states[static_cast<int>(keycode)] == KeyState::pressed;
}

bool Window::is_key_up(Window::KeyCode keycode)
{
    return key_states[static_cast<int>(keycode)] == KeyState::released;
}

bool Window::is_key_held(Window::KeyCode keycode)
{
    return key_states[static_cast<int>(keycode)] == KeyState::held ||
           key_states[static_cast<int>(keycode)] == KeyState::repeat;
}

void Window::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    auto window_ptr = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
    if (action == GLFW_RELEASE)
        window_ptr->key_states[key] = KeyState::released;
    else if (action == GLFW_REPEAT)
        window_ptr->key_states[key] = KeyState::repeat;
    else if (action == GLFW_PRESS)
        window_ptr->key_states[key] = KeyState::pressed;

    ImGuiIO& io = ImGui::GetIO();
    if (action == GLFW_PRESS) io.KeysDown[key] = true;
    if (action == GLFW_RELEASE) io.KeysDown[key] = false;

    // Modifiers are not reliable across systems
    io.KeyCtrl = io.KeysDown[GLFW_KEY_LEFT_CONTROL] || io.KeysDown[GLFW_KEY_RIGHT_CONTROL];
    io.KeyShift = io.KeysDown[GLFW_KEY_LEFT_SHIFT] || io.KeysDown[GLFW_KEY_RIGHT_SHIFT];
    io.KeyAlt = io.KeysDown[GLFW_KEY_LEFT_ALT] || io.KeysDown[GLFW_KEY_RIGHT_ALT];
#ifdef _WIN32
    io.KeySuper = false;
#else
    io.KeySuper = io.KeysDown[GLFW_KEY_LEFT_SUPER] || io.KeysDown[GLFW_KEY_RIGHT_SUPER];
#endif
}
void Window::char_callback(GLFWwindow* window, uint32_t codepoint)
{
    auto window_ptr = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
    ImGuiIO& io = ImGui::GetIO();
    io.AddInputCharacter(codepoint);
}

void Window::mouse_click_callback(GLFWwindow* window, int button, int action, int mods)
{
    Action callback_action;
    switch (action)
    {
        case GLFW_RELEASE:
            callback_action = Action::RELEASE;
            break;
        case GLFW_PRESS:
            callback_action = Action::PRESS;
            break;
        case GLFW_REPEAT:
            callback_action = Action::REPEAT;
            break;
        default:
            callback_action = Action::UNKNOWN;
    }

    Mouse mouse_button;
    switch (button)
    {
        case GLFW_MOUSE_BUTTON_1:
            mouse_button = Mouse::LEFT;
            break;
        case GLFW_MOUSE_BUTTON_2:
            mouse_button = Mouse::RIGHT;
            break;
        case GLFW_MOUSE_BUTTON_3:
            mouse_button = Mouse::MIDDLE;
            break;
        default:
            mouse_button = Mouse::OTHER;
    }

    auto window_ptr = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
    for (auto& callback : window_ptr->mouse_click_callbacks)
        callback(mouse_button, callback_action);

    if (!window_ptr->mouse_locked_to_window)
    {
        ImGuiIO& io = ImGui::GetIO();
        if (action < GLFW_REPEAT) io.MouseDown[button] = action;
    }
}

void Window::mouse_move_callback(GLFWwindow* window, double x, double y)
{
    auto window_ptr = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));

    window_ptr->mouse_position_x = x;
    window_ptr->mouse_position_y = y;

    window_ptr->mouse_change_in_previous_x =
        window_ptr->mouse_position_previous_x - window_ptr->mouse_position_x;
    window_ptr->mouse_change_in_previous_y =
        window_ptr->mouse_position_previous_y - window_ptr->mouse_position_y;

    // coordinates are reversed on y axis (top left vs bottom left)
    window_ptr->mouse_change_in_previous_x *= -1;

    window_ptr->mouse_position_previous_x = window_ptr->mouse_position_x;
    window_ptr->mouse_position_previous_y = window_ptr->mouse_position_y;

    for (auto& callback : window_ptr->mouse_move_callbacks)
        callback(window_ptr->mouse_change_in_previous_x, window_ptr->mouse_change_in_previous_y);

    ImGuiIO& io = ImGui::GetIO();
    io.MousePos = ImVec2(static_cast<float>(window_ptr->mouse_position_x),
                         static_cast<float>(window_ptr->mouse_position_y));
}

void Window::scroll_callback(GLFWwindow* window, double x, double y)
{
    auto window_ptr = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
    for (auto& callback : window_ptr->scroll_callbacks) callback(x, y);

    ImGuiIO& io = ImGui::GetIO();
    io.MouseWheelH += static_cast<float>(x);
    io.MouseWheel += static_cast<float>(y);
}

Window::Settings Window::get_settings() { return active_settings; }

GLFWwindow* Window::get_window_pointer() { return window_ptr; }

bool Window::should_close() { return glfwWindowShouldClose(window_ptr); }

void Window::set_close() { glfwSetWindowShouldClose(window_ptr, GLFW_TRUE); }

bool Window::is_mouse_locked_to_window() { return mouse_locked_to_window; }
void Window::set_mouse_window_lock(bool locked)
{
    mouse_locked_to_window = locked;
    glfwSetInputMode(window_ptr, GLFW_CURSOR,
                     mouse_locked_to_window ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    if (locked)
        glfwGetCursorPos(window_ptr, &last_mouse_position_x, &last_mouse_position_y);
    else
        glfwSetCursorPos(window_ptr, last_mouse_position_x, last_mouse_position_y);
}