#include "rvpt.h"

#include <cstdlib>

#include <algorithm>
#include <fstream>
#include <iostream>

#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <nlohmann/json.hpp>
#include <imgui.h>
#include <fmt/core.h>

#include "imgui_helpers.h"
#include "imgui_internal.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

struct DebugVertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 color;
};

/*
uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
    {
        if ((typeFilter & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type!");
}

void createBuffer(VkPhysicalDevice physicalDevice, VkDevice device, VkDeviceSize size,
                  VkBufferUsageFlags usage,
                  VkMemoryPropertyFlags properties,
                  VkBuffer& buffer, VkDeviceMemory& bufferMemory)
{
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling,
                 VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image,
                 VkDeviceMemory& imageMemory, VkPhysicalDevice physicalDevice, VkDevice device)
{
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create image!");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, image, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to allocate image memory!");
    }

    vkBindImageMemory(device, image, imageMemory, 0);
}

void RVPT::createTextureImage(VkPhysicalDevice physicalDevice, VkImage textureImage,
                        VkDeviceMemory textureImageMemory)
{ 
    int texWidth, texHeight, texChannels; 
    stbi_uc* pixels =
    stbi_load("../img/statue.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

    VkDeviceSize imageSize = texWidth * texHeight * 4;

    if (!pixels)
    {
        throw std::runtime_error("failed to load texture image!");
    }

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(physicalDevice, vk_device, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer, stagingBufferMemory);

    void* data;
    vkMapMemory(vk_device, stagingBufferMemory, 0, imageSize, 0, &data);
    memcpy(data, pixels, static_cast<size_t>(imageSize));
    vkUnmapMemory(vk_device, stagingBufferMemory);

    stbi_image_free(pixels);

    createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory,
                physicalDevice, vk_device);

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;

    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;

    region.imageOffset = {0, 0, 0};
    region.imageExtent = {texWidth, texHeight, 1};

    VK::CommandBuffer commandBuffer =
        VK::CommandBuffer(vk_device, compute_queue.has_value() ? *compute_queue : *graphics_queue,
                          "texture_command_buffer_");
    
    vkCmdCopyBufferToImage(commandBuffer.get(), 
                            stagingBuffer, 
                            textureImage, 
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 
                            1,
                           &region);

    vkDestroyBuffer(vk_device, stagingBuffer, nullptr);
    vkFreeMemory(vk_device, stagingBufferMemory, nullptr);
} */

bool RVPT::PreviousFrameState::operator==(RVPT::PreviousFrameState const& right)
{
    return settings.render_mode == right.settings.render_mode &&
           settings.camera_mode == right.settings.camera_mode && camera_data == right.camera_data &&
           settings.scene == right.settings.scene;
}

RVPT::RVPT(Window& window)
    : window_ref(window),
      scene_camera(window.get_aspect_ratio(), glm::vec3(1.5, 2, -2), glm::vec3(-38, 36, 0)),
      random_generator(std::random_device{}()),
      distribution(0.0f, 1.0f)
{
    ImGui::CreateContext();

    std::ifstream input("project_configuration.json");
    nlohmann::json json;
    input >> json;
    if (json.contains("project_source_dir"))
    {
        source_folder = json["project_source_dir"];
    }

    random_numbers.resize(20480);

}

RVPT::~RVPT() {}

bool RVPT::initialize()
{
    bool init = context_init();
    pipeline_builder = VK::PipelineBuilder(vk_device, source_folder);
    memory_allocator =
        VK::MemoryAllocator(context.device.physical_device.physical_device, vk_device);

    init &= swapchain_init();
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        sync_resources.emplace_back(vk_device, graphics_queue.value(), present_queue.value(),
                                    vkb_swapchain.swapchain);
    }
    frames_inflight_fences.resize(vkb_swapchain.image_count, nullptr);

    fullscreen_tri_render_pass = VK::create_render_pass(
        vk_device, vkb_swapchain.image_format,
        VK::get_depth_image_format(context.device.physical_device.physical_device),
        "fullscreen_image_copy_render_pass");

    imgui_impl.emplace(vk_device, *graphics_queue, pipeline_builder, memory_allocator,
                       fullscreen_tri_render_pass, vkb_swapchain.extent, MAX_FRAMES_IN_FLIGHT);

    rendering_resources = create_rendering_resources();

    create_framebuffers();

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        add_per_frame_data(i);
    }

    return init;
}
bool RVPT::update()
{
    render_settings.visualize_probes = visualize_probes;

    auto camera_data = scene_camera.get_data();

    render_settings.camera_mode = scene_camera.get_camera_mode();

    if (!(previous_frame_state == RVPT::PreviousFrameState{render_settings, camera_data}))
    {
        //render_settings.current_frame = 0;
        previous_frame_state.settings = render_settings;
        previous_frame_state.camera_data = camera_data;
    }
    else
    {
        //render_settings.current_frame++;
    }

    for (auto& r : random_numbers) r = (distribution(random_generator));

    per_frame_data[current_frame_index].raytrace_work_fence.wait();
    per_frame_data[current_frame_index].raytrace_work_fence.reset();

    per_frame_data[current_frame_index].settings_uniform.copy_to(render_settings);
    per_frame_data[current_frame_index].random_buffer.copy_to(random_numbers);
    per_frame_data[current_frame_index].camera_uniform.copy_to(camera_data);

    float delta = static_cast<float>(time.since_last_frame());

    per_frame_data[current_frame_index].sphere_buffer.copy_to(spheres);
    per_frame_data[current_frame_index].probe_buffer.copy_to(probe_rays);
	
    per_frame_data[current_frame_index].irradiance_field_uniform.copy_to(ir);

    return true;
}

void RVPT::update_imgui()
{
    if (!show_imgui) return;

    ImGuiIO& io = ImGui::GetIO();

    // Setup display size (every frame to accommodate for window resizing)
    int w, h;
    int display_w, display_h;
    auto win_ptr = window_ref.get_window_pointer();
    glfwGetWindowSize(win_ptr, &w, &h);
    glfwGetFramebufferSize(win_ptr, &display_w, &display_h);
    io.DisplaySize = ImVec2((float)w, (float)h);
    if (w > 0 && h > 0)
        io.DisplayFramebufferScale = ImVec2((float)display_w / w, (float)display_h / h);

    // Setup time step
    io.DeltaTime = static_cast<float>(time.since_last_frame());

    // imgui back end can't show 2 windows
    static bool show_stats = true;
    ImGui::SetNextWindowPos({0, 0}, ImGuiCond_Once);
    ImGui::SetNextWindowSize({200, 65}, ImGuiCond_Once);
    if (ImGui::Begin("Stats", &show_stats))
    {
        ImGui::Text("Frame Time %.4f", time.average_frame_time());
        ImGui::Text("FPS %.2f", 1.0 / time.average_frame_time());
    }
    ImGui::End();
    static bool show_render_settings = true;
    ImGui::SetNextWindowPos({0, 65}, ImGuiCond_Once);
    ImGui::SetNextWindowSize({200, 360}, ImGuiCond_Once);
    if (ImGui::Begin("Render Settings", &show_stats))
    {
        ImGui::PushItemWidth(80);
        ImGui::SliderInt("Scene", &render_settings.scene, 0, 2);
        ImGui::Text("Render Mode");
        ImGui::PushItemWidth(0);
        dropdown_helper("render_mode", render_settings.render_mode, RenderModes);

        ImGui::Checkbox("Visualize Probes", &visualize_probes);
        ImGui::Text("Number of Probes");
        if (ImGui::SliderInt("X", &ir.probe_count.x, 1, 19) ||
            ImGui::SliderInt("Y", &ir.probe_count.y, 1, 19) ||
            ImGui::SliderInt("Z", &ir.probe_count.z, 1, 19) )
        {
            need_change_probe_texture = true;
        }
        ImGui::Text("Number of Probe Rays");

        if (ImGui::SliderInt("(sqrt)", &ir.sqrt_rays_per_probe, 2, 30))
        {
            need_change_probe_texture = true;
        }
        ImGui::Text("Probe Distance");
        if (ImGui::SliderInt("dist", &ir.side_length, 2, 15))
        {
            need_generate_probe_rays = true;
        }

        ImGui::Text("Field Origin");
        ImGui::PushItemWidth(125);
        if (ImGui::DragFloat3("ori", glm::value_ptr(ir.field_origin), 0.2))
        {
            need_generate_probe_rays = true;
        }

        if (need_change_probe_texture || need_generate_probe_rays)
        {
            if (ImGui::Button("Recalculate Probes", ImVec2(175, 30)))
            {
                recreate_probe_textures();
            }
        }
    }
    ImGui::End();

    scene_camera.update_imgui();
}

RVPT::draw_return RVPT::draw()
{
    time.frame_start();

    record_compute_command_buffer();

    VK::Queue& compute_submit = compute_queue.has_value() ? *compute_queue : *graphics_queue;
    compute_submit.submit(per_frame_data[current_frame_index].raytrace_command_buffer,
                          per_frame_data[current_frame_index].raytrace_work_fence);

    auto& current_frame = sync_resources[current_sync_index];

    current_frame.command_fence.wait();
    current_frame.command_buffer.reset();

    uint32_t swapchain_image_index;
    VkResult result = vkAcquireNextImageKHR(vk_device, vkb_swapchain.swapchain, UINT64_MAX,
                                            current_frame.image_avail_sem.get(), VK_NULL_HANDLE,
                                            &swapchain_image_index);

    if (result == VK_ERROR_OUT_OF_DATE_KHR)
    {
        swapchain_reinit();
        return draw_return::swapchain_out_of_date;
    }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
    {
        fmt::print(stderr, "Failed to acquire next swapchain image\n");
        assert(false);
    }
    record_command_buffer(current_frame, swapchain_image_index);

    if (frames_inflight_fences[swapchain_image_index] != nullptr)
    {
        vkWaitForFences(vk_device, 1, &frames_inflight_fences[swapchain_image_index], VK_TRUE,
                        UINT64_MAX);
    }
    frames_inflight_fences[swapchain_image_index] = current_frame.command_fence.get();

    current_frame.command_fence.reset();
    current_frame.submit();

    result = current_frame.present(swapchain_image_index);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebuffer_resized)
    {
        framebuffer_resized = false;
        swapchain_reinit();
    }
    else if (result != VK_SUCCESS)
    {
        fmt::print(stderr, "Failed to present swapchain image\n");
        assert(false);
    }
    current_sync_index = (current_sync_index + 1) % sync_resources.size();
    current_frame_index = (current_frame_index + 1) % per_frame_data.size();

    time.frame_stop();
    return draw_return::success;
}

void RVPT::shutdown()
{
    if (compute_queue) compute_queue->wait_idle();
    graphics_queue->wait_idle();
    present_queue->wait_idle();

    per_frame_data.clear();
    rendering_resources.reset();

    imgui_impl.reset();

    VK::destroy_render_pass(vk_device, fullscreen_tri_render_pass);

    framebuffers.clear();

    sync_resources.clear();
    vkb_swapchain.destroy_image_views(swapchain_image_views);

    memory_allocator.shutdown();
    pipeline_builder.shutdown();
    vkb::destroy_swapchain(vkb_swapchain);
    vkb::destroy_device(context.device);
    vkDestroySurfaceKHR(context.inst.instance, context.surf, nullptr);
    vkb::destroy_instance(context.inst);
}

void replace_all(std::string& str, const std::string& from, const std::string& to)
{
    if (from.empty()) return;
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos)
    {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();  // In case 'to' contains 'from', like replacing 'x' with 'yx'
    }
}

void RVPT::reload_shaders()
{
    if (source_folder == "")
    {
        fmt::print("source_folder not set, unable to reload shaders\n");
        return;
    }
    fmt::print("Compiling Shaders:\n");
#ifdef WIN32
    auto double_backslash = source_folder;
    replace_all(double_backslash, "/", "\\\\");
    std::string str = fmt::format(
        "cd {0}\\\\assets\\\\shaders && {0}\\\\scripts\\\\compile_shaders.bat", double_backslash);
    std::system(str.c_str());
#elif __unix__
    std::string str =
        fmt::format("cd {0}/assets/shaders && bash {0}/scripts/compile_shaders.sh", source_folder);
    std::system(str.c_str());
#endif
    if (compute_queue) compute_queue->wait_idle();
    graphics_queue->wait_idle();
    present_queue->wait_idle();

    pipeline_builder.recompile_pipelines();
}

void RVPT::set_raytrace_mode(int mode) { render_settings.render_mode = mode; }

// Private functions //
bool RVPT::context_init()
{
#if defined(DEBUG) || defined(_DEBUG)
    bool use_validation = true;
#else
    bool use_validation = false;
#endif

    vkb::InstanceBuilder inst_builder;
    auto inst_ret =
        inst_builder.set_app_name(window_ref.get_settings().title)
            .request_validation_layers(use_validation)
            .set_debug_callback([](VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                   VkDebugUtilsMessageTypeFlagsEXT messageType,
                                   const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                   void* pUserData) -> VkBool32 {
                auto severity = vkb::to_string_message_severity(messageSeverity);
                auto type = vkb::to_string_message_type(messageType);
                fmt::print("[{}: {}] {}\n", severity, type, pCallbackData->pMessage);
                return VK_FALSE;
            })
            .build();

    if (!inst_ret)
    {
        fmt::print(stderr, "Failed to create an instance: {}\n", inst_ret.error().message());
        return false;
    }

    context.inst = inst_ret.value();

    VkResult surf_res = glfwCreateWindowSurface(
        context.inst.instance, window_ref.get_window_pointer(), nullptr, &context.surf);
    if (surf_res != VK_SUCCESS)
    {
        fmt::print(stderr, "Failed to create a surface: Error Code{}\n", surf_res);
        return false;
    }
    VkPhysicalDeviceFeatures required_features{};
    required_features.samplerAnisotropy = true;
    required_features.fillModeNonSolid = true;

    vkb::PhysicalDeviceSelector selector(context.inst);
    auto phys_ret = selector.set_surface(context.surf)
                        .set_required_features(required_features)
                        .set_minimum_version(1, 1)
                        .select();

    if (!phys_ret)
    {
        fmt::print(stderr, "Failed to find a physical device: \n", phys_ret.error().message());
        return false;
    }

    vkb::DeviceBuilder dev_builder(phys_ret.value());
    auto dev_ret = dev_builder.build();
    if (!dev_ret)
    {
        fmt::print(stderr, "Failed create a device: \n", dev_ret.error().message());
        return false;
    }

    context.device = dev_ret.value();
    vk_device = dev_ret.value().device;

    auto graphics_queue_index_ret = context.device.get_queue_index(vkb::QueueType::graphics);
    if (!graphics_queue_index_ret)
    {
        fmt::print(stderr, "Failed to get the graphics queue: \n",
                   graphics_queue_index_ret.error().message());
        return false;
    }
    graphics_queue.emplace(vk_device, graphics_queue_index_ret.value(), "graphics_queue");

    auto present_queue_index_ret = context.device.get_queue_index(vkb::QueueType::present);
    if (!present_queue_index_ret)
    {
        fmt::print(stderr, "Failed to get the present queue: \n",
                   present_queue_index_ret.error().message());
        return false;
    }
    present_queue.emplace(vk_device, present_queue_index_ret.value(), "present_queue");

    auto compute_queue_index_ret =
        context.device.get_dedicated_queue_index(vkb::QueueType::compute);
    if (compute_queue_index_ret)
    {
        compute_queue.emplace(vk_device, compute_queue_index_ret.value(), "compute_queue");
    }

    VK::setup_debug_util_helper(vk_device);

    return true;
}

bool RVPT::swapchain_init()
{
    vkb::SwapchainBuilder swapchain_builder(context.device);
    auto ret = swapchain_builder.build();
    if (!ret)
    {
        fmt::print(stderr, "Failed to create a swapchain: \n", ret.error().message());
        return false;
    }
    vkb_swapchain = ret.value();
    return swapchain_get_images();
}

bool RVPT::swapchain_reinit()
{
    framebuffers.clear();
    vkb_swapchain.destroy_image_views(swapchain_image_views);

    vkb::SwapchainBuilder swapchain_builder(context.device);
    auto ret = swapchain_builder.set_old_swapchain(vkb_swapchain).build();
    if (!ret)
    {
        fmt::print(stderr, "Failed to recreate a swapchain: \n", ret.error().message());
        return false;
    }
    vkb::destroy_swapchain(vkb_swapchain);
    vkb_swapchain = ret.value();
    bool out_bool = swapchain_get_images();
    create_framebuffers();
    return out_bool;
}

bool RVPT::swapchain_get_images()
{
    auto swapchain_images_ret = vkb_swapchain.get_images();
    if (!swapchain_images_ret)
    {
        return false;
    }
    swapchain_images = swapchain_images_ret.value();

    auto swapchain_image_views_ret = vkb_swapchain.get_image_views();
    if (!swapchain_image_views_ret)
    {
        return false;
    }
    swapchain_image_views = swapchain_image_views_ret.value();

    return true;
}

void RVPT::create_framebuffers()
{
    framebuffers.clear();
    for (uint32_t i = 0; i < vkb_swapchain.image_count; i++)
    {
        std::vector<VkImageView> image_views = {
            swapchain_image_views[i], rendering_resources->depth_buffer.image_view.handle};
        VK::debug_utils_helper.set_debug_object_name(VK_OBJECT_TYPE_IMAGE_VIEW,
                                                     swapchain_image_views[i],
                                                     "swapchain_image_view_" + std::to_string(i));

        framebuffers.emplace_back(vk_device, fullscreen_tri_render_pass, vkb_swapchain.extent,
                                  image_views, "swapchain_framebuffer_" + std::to_string(i));
    }
}

VK::Image RVPT::create_probe_texture_albedo() {
    int probe_texture_width = ir.probe_count.x * ir.probe_count.z * ir.sqrt_rays_per_probe;
    int probe_texture_height = ir.probe_count.y * ir.sqrt_rays_per_probe;

    auto probe_texture_albedo =
        VK::Image(vk_device, memory_allocator, *graphics_queue, "probe_texture_albedo",
                  VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, probe_texture_width,
                  probe_texture_height, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                  VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT,
                  static_cast<VkDeviceSize>(probe_texture_width * probe_texture_height * 4),
                  VK::MemoryUsage::gpu);
    return std::move(probe_texture_albedo);
}

void RVPT::recreate_probe_textures() {
    for (int i = 0; i < per_frame_data.size(); i++)
    {
        const VkFence& f = per_frame_data[i].probe_work_fence.get();
        const VkFence& f2 = per_frame_data[i].raytrace_work_fence.get();

        vkWaitForFences(vk_device, 1, &f, VK_TRUE, 3e9);
        vkWaitForFences(vk_device, 1, &f2, VK_TRUE, 3e9);
    }

    if (need_change_probe_texture)
    {
       // Currently this causes crashes, but we need to properly delete the image 
       // before creating the next ones...
       /* rendering_resources->probe_texture_albedo.sampler.~HandleWrapper();
          rendering_resources->probe_texture_albedo.image_view.~HandleWrapper();
          rendering_resources->probe_texture_albedo.image_allocation.~Allocation();
          rendering_resources->probe_texture_albedo.image.~HandleWrapper();*/

        int probe_texture_width = ir.probe_count.x * ir.probe_count.z * ir.sqrt_rays_per_probe;
        int probe_texture_height = ir.probe_count.y * ir.sqrt_rays_per_probe;

        rendering_resources->probe_texture_albedo =
            VK::Image(vk_device, memory_allocator, *graphics_queue, "probe_texture_albedo",
                      VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, probe_texture_width,
                      probe_texture_height, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                      VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT,
                      static_cast<VkDeviceSize>(probe_texture_width * probe_texture_height * 4),
                      VK::MemoryUsage::gpu);

         rendering_resources->probe_texture_distance =
            VK::Image(vk_device, memory_allocator, *graphics_queue, "probe_texture_distance",
                      VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, probe_texture_width,
                      probe_texture_height, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                      VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT,
                      static_cast<VkDeviceSize>(probe_texture_width * probe_texture_height * 4),
                      VK::MemoryUsage::gpu);

        for (int i = 0; i < per_frame_data.size(); i++)
        {
            auto &frame = per_frame_data[i];

            
            std::vector<VK::DescriptorUseVector> raytracing_descriptors;
            raytracing_descriptors.push_back(std::vector{frame.settings_uniform.descriptor_info()});
            raytracing_descriptors.push_back(std::vector{frame.output_image.descriptor_info()});
            raytracing_descriptors.push_back(std::vector{frame.random_buffer.descriptor_info()});
            raytracing_descriptors.push_back(std::vector{frame.camera_uniform.descriptor_info()});
            raytracing_descriptors.push_back(std::vector{frame.sphere_buffer.descriptor_info()});

            raytracing_descriptors.push_back(
                std::vector{rendering_resources->probe_texture_albedo.descriptor_info()});
            raytracing_descriptors.push_back(
                std::vector{rendering_resources->probe_texture_distance.descriptor_info()});
            // S_CHANGED
            raytracing_descriptors.push_back(
                std::vector{frame.irradiance_field_uniform.descriptor_info()});
            raytracing_descriptors.push_back(std::vector{
                rendering_resources->block_texture.descriptor_info()});  // HELEN: CHANGED THIS

            rendering_resources->raytrace_descriptor_pool.update_descriptor_sets(
                frame.raytracing_descriptor_sets, raytracing_descriptors);

        }

        need_generate_probe_rays = true;
    }

    if (need_generate_probe_rays)
    {
        generate_probe_rays();

        if (need_change_probe_texture)
        {
            for (int i = 0; i < per_frame_data.size(); i++)
            {
                auto& frame = per_frame_data[i];
                // vkDestroyBuffer(vk_device, frame.probe_buffer.get(), NULL);
                frame.probe_buffer =
                    VK::Buffer(vk_device, memory_allocator, "probes_buffer_" + std::to_string(i),
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               sizeof(ProbeRay) * probe_rays.size(), VK::MemoryUsage::cpu_to_gpu);

                std::vector<VK::DescriptorUseVector> probe_descriptors;
                probe_descriptors.push_back(std::vector{frame.settings_uniform.descriptor_info()});
                probe_descriptors.push_back(std::vector{frame.probe_buffer.descriptor_info()});
                probe_descriptors.push_back(
                    std::vector{rendering_resources->probe_texture_albedo.descriptor_info()});
                probe_descriptors.push_back(
                    std::vector{rendering_resources->probe_texture_distance.descriptor_info()});
                probe_descriptors.push_back(std::vector{frame.sphere_buffer.descriptor_info()});
                probe_descriptors.push_back(
                    std::vector{frame.irradiance_field_uniform.descriptor_info()});
                rendering_resources->probe_descriptor_pool.update_descriptor_sets(
                    frame.probe_descriptor_sets, probe_descriptors);
            }
        }
    }

    need_change_probe_texture = false;
}

RVPT::RenderingResources RVPT::create_rendering_resources()
{
    std::vector<VkDescriptorSetLayoutBinding> layout_bindings = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}};

    auto image_pool = VK::DescriptorPool(vk_device, layout_bindings, MAX_FRAMES_IN_FLIGHT * 2,
                                         "image_descriptor_pool");

    auto fullscreen_triangle_pipeline_layout = pipeline_builder.create_layout(
        {image_pool.layout()}, {}, "fullscreen_triangle_pipeline_layout");

    VK::GraphicsPipelineDetails fullscreen_details;
    fullscreen_details.name = "fullscreen_pipeline";
    fullscreen_details.pipeline_layout = fullscreen_triangle_pipeline_layout;
    fullscreen_details.vert_shader = "fullscreen_tri.vert.spv";
    fullscreen_details.frag_shader = "tex_sample.frag.spv";
    fullscreen_details.render_pass = fullscreen_tri_render_pass;
    fullscreen_details.extent = vkb_swapchain.extent;

    auto fullscreen_triangle_pipeline = pipeline_builder.create_pipeline(fullscreen_details);
    std::vector<VkDescriptorSetLayoutBinding> compute_layout_bindings = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // render settings
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,   1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // result image
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // random numbers
        {3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // camera
        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // spheres
        {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,   1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // probe texture (albedo)
        {6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // probe texture (distances)
		{7, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // irradiance field info
        {8, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr} // HELEN: ADDED TEXTURE
    };

    /* LOOK: Add stuff here if you want to add more variables to the probe pass shader.
             There are different descriptor types:
                   UNIFORM_BUFFER -> for uniforms
                   STORAGE_BUFFER -> you can write to these, but here they act as uniforms. that's how original code passes in spheres / triangles
                   STORAGE_IMAGE  -> textures */

    std::vector<VkDescriptorSetLayoutBinding> probe_layout_bindings = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // render settings
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // rays
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // output albedo
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // output distance
        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // spheres
        {5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}  // irradiance field info
    };

    // Probe pipeline setup
    auto probe_descriptor_pool = VK::DescriptorPool(
        vk_device, probe_layout_bindings, MAX_FRAMES_IN_FLIGHT, "probe_descriptor_pool");

    auto probe_pipeline_layout = pipeline_builder.create_layout(
        {probe_descriptor_pool.layout()}, {}, "probe_pipeline_layout");

    VK::ComputePipelineDetails probe_details;
    probe_details.name = "probe_compute_pipeline";
    probe_details.pipeline_layout = probe_pipeline_layout;
    probe_details.compute_shader = "probe_pass.comp.spv";

    auto probe_pipeline = pipeline_builder.create_pipeline(probe_details);

    // Raytrace pipeline setup
    auto raytrace_descriptor_pool = VK::DescriptorPool(
        vk_device, compute_layout_bindings, MAX_FRAMES_IN_FLIGHT, "raytrace_descriptor_pool");  

    auto raytrace_pipeline_layout = pipeline_builder.create_layout(
        {raytrace_descriptor_pool.layout()}, {}, "raytrace_pipeline_layout");

    VK::ComputePipelineDetails raytrace_details;
    raytrace_details.name = "raytrace_compute_pipeline";
    raytrace_details.pipeline_layout = raytrace_pipeline_layout;
    raytrace_details.compute_shader = "compute_pass.comp.spv";

    auto raytrace_pipeline = pipeline_builder.create_pipeline(raytrace_details);

    std::vector<VkDescriptorSetLayoutBinding> debug_layout_bindings = {
        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr}};

    auto debug_descriptor_pool = VK::DescriptorPool(vk_device, debug_layout_bindings,
                                                    MAX_FRAMES_IN_FLIGHT, "debug_descriptor_pool");
    auto debug_pipeline_layout = pipeline_builder.create_layout({debug_descriptor_pool.layout()},
                                                                {}, "debug_vis_pipeline_layout");

    std::vector<VkVertexInputBindingDescription> binding_desc = {
        {0, sizeof(DebugVertex), VK_VERTEX_INPUT_RATE_VERTEX}};

    std::vector<VkVertexInputAttributeDescription> attribute_desc = {
        {0, binding_desc[0].binding, VK_FORMAT_R32G32B32_SFLOAT, 0},
        {1, binding_desc[0].binding, VK_FORMAT_R32G32B32_SFLOAT, 12},
        {2, binding_desc[0].binding, VK_FORMAT_R32G32B32_SFLOAT, 24}};

    VkPipelineDepthStencilStateCreateInfo depth_stencil_info{};
    depth_stencil_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

    VK::GraphicsPipelineDetails debug_details;
    debug_details.name = "debug_raster_view_pipeline";
    debug_details.pipeline_layout = debug_pipeline_layout;
    debug_details.vert_shader = "debug_vis.vert.spv";
    debug_details.frag_shader = "debug_vis.frag.spv";
    debug_details.render_pass = fullscreen_tri_render_pass;
    debug_details.extent = vkb_swapchain.extent;
    debug_details.binding_desc = binding_desc;
    debug_details.attribute_desc = attribute_desc;
    debug_details.cull_mode = VK_CULL_MODE_NONE;
    debug_details.enable_depth = true;

    auto opaque = pipeline_builder.create_pipeline(debug_details);
    debug_details.polygon_mode = VK_POLYGON_MODE_LINE;
    auto wireframe = pipeline_builder.create_pipeline(debug_details);

    int probe_texture_width = ir.probe_count.x * ir.probe_count.z * ir.sqrt_rays_per_probe;
    int probe_texture_height = ir.probe_count.y * ir.sqrt_rays_per_probe;

    /* LOOK: These are the actual definitions of the textures using VK::Image */
    auto probe_texture_albedo = create_probe_texture_albedo();

    auto probe_texture_distance = VK::Image(
        vk_device,
        memory_allocator,
        *graphics_queue,
        "probe_texture",
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_TILING_OPTIMAL,
        probe_texture_width,
        probe_texture_height,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_ASPECT_COLOR_BIT,
        static_cast<VkDeviceSize>(probe_texture_width * probe_texture_height * 4),
        VK::MemoryUsage::gpu
    );

    // HELEN: ADDED THIS
    auto block_texture = VK::Image(
        vk_device, 
        memory_allocator, 
        *graphics_queue, 
        "block_texture",
        VK_FORMAT_R8G8B8A8_UNORM, 
        VK_IMAGE_TILING_OPTIMAL, 
        512,
        512, 
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        VK_IMAGE_LAYOUT_GENERAL, 
        VK_IMAGE_ASPECT_COLOR_BIT,
        static_cast<VkDeviceSize>(512 * 512 * 4),
        VK::MemoryUsage::gpu
    );
    VkImage textureImage;
    VkDeviceMemory textureImageMemory;
    //createTextureImage(context.device.physical_device.physical_device, textureImage, text);

    VkFormat depth_format =
        VK::get_depth_image_format(context.device.physical_device.physical_device);

    auto depth_image =
        VK::Image(vk_device, memory_allocator, *graphics_queue, "depth_image", depth_format,
                  VK_IMAGE_TILING_OPTIMAL, window_ref.get_settings().width,
                  window_ref.get_settings().height, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                  VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                  VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
                  static_cast<VkDeviceSize>(window_ref.get_settings().width *
                                            window_ref.get_settings().height * 4),
                  VK::MemoryUsage::gpu);

    return RVPT::RenderingResources{std::move(image_pool),
                                    std::move(raytrace_descriptor_pool),
                                    std::move(probe_descriptor_pool),
                                    std::move(debug_descriptor_pool),
                                    fullscreen_triangle_pipeline_layout,
                                    fullscreen_triangle_pipeline,
                                    probe_pipeline_layout,
                                    probe_pipeline,
                                    raytrace_pipeline_layout,
                                    raytrace_pipeline,
                                    debug_pipeline_layout,
                                    opaque,
                                    wireframe,
                                    std::move(probe_texture_albedo),
                                    std::move(probe_texture_distance),
                                    std::move(block_texture), // HELEN: ADDED THIS
                                    std::move(depth_image)};
}

void RVPT::add_per_frame_data(int index)
{
    auto settings_uniform = VK::Buffer(
        vk_device, memory_allocator, "settings_buffer_" + std::to_string(index),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sizeof(RenderSettings), VK::MemoryUsage::cpu_to_gpu);
    auto output_image = VK::Image(vk_device, memory_allocator, *graphics_queue,
                                  "raytrace_output_image_" + std::to_string(index),
                                  VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL,
                                  window_ref.get_settings().width, window_ref.get_settings().height,
                                  VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                                  VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT,
                                  static_cast<VkDeviceSize>(window_ref.get_settings().width *
                                                            window_ref.get_settings().height * 4),
                                  VK::MemoryUsage::gpu);
    auto random_buffer =
        VK::Buffer(vk_device, memory_allocator, "random_data_uniform_" + std::to_string(index),
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                   sizeof(decltype(random_numbers)::value_type) * random_numbers.size(),
                   VK::MemoryUsage::cpu_to_gpu);

    auto temp_camera_data = scene_camera.get_data();
    auto camera_uniform =
        VK::Buffer(vk_device, memory_allocator, "camera_uniform_" + std::to_string(index),
                   VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                   sizeof(decltype(temp_camera_data)::value_type) * temp_camera_data.size(),
                   VK::MemoryUsage::cpu_to_gpu);
    auto sphere_buffer =
        VK::Buffer(vk_device, memory_allocator, "spheres_buffer_" + std::to_string(index),
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, sizeof(Sphere) * spheres.size(),
                   VK::MemoryUsage::cpu_to_gpu);

    // LOOK: the definition of the probe ray buffer as well as command buffer + workfence
    auto probe_buffer =
        VK::Buffer(vk_device, memory_allocator, "probes_buffer_" + std::to_string(index),
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, sizeof(ProbeRay) * probe_rays.size(),
                   VK::MemoryUsage::cpu_to_gpu);

    auto probe_command_buffer =
        VK::CommandBuffer(vk_device, compute_queue.has_value() ? *compute_queue : *graphics_queue,
                          "probe_command_buffer_" + std::to_string(index));
    auto probe_work_fence = VK::Fence(vk_device, "probe_work_fence_" + std::to_string(index));

	// S_CHANGED
    auto irradiance_field_uniform =
        VK::Buffer(vk_device, memory_allocator, "irradiance_field_uniform_" + std::to_string(index),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sizeof(IrradianceField), VK::MemoryUsage::cpu_to_gpu);

    auto raytrace_command_buffer =
        VK::CommandBuffer(vk_device, compute_queue.has_value() ? *compute_queue : *graphics_queue,
                          "raytrace_command_buffer_" + std::to_string(index));
    auto raytrace_work_fence = VK::Fence(vk_device, "raytrace_work_fence_" + std::to_string(index));

    // descriptor sets
    auto image_descriptor_set = rendering_resources->image_pool.allocate(
        "output_image_descriptor_set_" + std::to_string(index));
    auto raytracing_descriptor_set = rendering_resources->raytrace_descriptor_pool.allocate(
        "raytrace_descriptor_set_" + std::to_string(index));
    auto probe_descriptor_set = rendering_resources->probe_descriptor_pool.allocate(
        "probe_descriptor_set_" + std::to_string(index));

    // update descriptor sets with resources
    std::vector<VK::DescriptorUseVector> image_descriptors;
    image_descriptors.push_back(std::vector{output_image.descriptor_info()});
    rendering_resources->image_pool.update_descriptor_sets(image_descriptor_set, image_descriptors);

    std::vector<VK::DescriptorUseVector> raytracing_descriptors;
    raytracing_descriptors.push_back(std::vector{settings_uniform.descriptor_info()});
    raytracing_descriptors.push_back(std::vector{output_image.descriptor_info()});
    raytracing_descriptors.push_back(std::vector{random_buffer.descriptor_info()});
    raytracing_descriptors.push_back(std::vector{camera_uniform.descriptor_info()});
    raytracing_descriptors.push_back(std::vector{sphere_buffer.descriptor_info()});
	
    raytracing_descriptors.push_back(
        std::vector{rendering_resources->probe_texture_albedo.descriptor_info()});
    raytracing_descriptors.push_back(
        std::vector{rendering_resources->probe_texture_distance.descriptor_info()});
    // S_CHANGED
    raytracing_descriptors.push_back(std::vector{irradiance_field_uniform.descriptor_info()});
    raytracing_descriptors.push_back(
        std::vector{rendering_resources->block_texture.descriptor_info()}); // HELEN: CHANGED THIS

    rendering_resources->raytrace_descriptor_pool.update_descriptor_sets(raytracing_descriptor_set,
                                                                         raytracing_descriptors);

    // LOOK: need to update this if you update the {0, 1, 2} thing referenced earlier
    std::vector<VK::DescriptorUseVector> probe_descriptors;
    probe_descriptors.push_back(std::vector{settings_uniform.descriptor_info()});
    probe_descriptors.push_back(std::vector{probe_buffer.descriptor_info()});
    probe_descriptors.push_back(std::vector{rendering_resources->probe_texture_albedo.descriptor_info()});
    probe_descriptors.push_back(std::vector{rendering_resources->probe_texture_distance.descriptor_info()});
    probe_descriptors.push_back(std::vector{sphere_buffer.descriptor_info()});
    probe_descriptors.push_back(std::vector{irradiance_field_uniform.descriptor_info()});
    rendering_resources->probe_descriptor_pool.update_descriptor_sets(probe_descriptor_set,
                                                                      probe_descriptors);

    // Debug vis
    auto debug_camera_uniform = VK::Buffer(
        vk_device, memory_allocator, "debug_camera_uniform_" + std::to_string(index),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sizeof(glm::mat4), VK::MemoryUsage::cpu_to_gpu);
    auto debug_vertex_buffer = VK::Buffer(
        vk_device, memory_allocator, "debug_vertecies_buffer_" + std::to_string(index),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, 1000 * sizeof(DebugVertex), VK::MemoryUsage::cpu_to_gpu);

    auto debug_descriptor_set = rendering_resources->debug_descriptor_pool.allocate(
        "debug_descriptor_set_" + std::to_string(index));

    std::vector<VK::DescriptorUseVector> debug_descriptors;
    debug_descriptors.push_back(std::vector{debug_camera_uniform.descriptor_info()});
    rendering_resources->debug_descriptor_pool.update_descriptor_sets(debug_descriptor_set,
                                                                      debug_descriptors);

    per_frame_data.push_back(RVPT::PerFrameData{
        std::move(settings_uniform), std::move(output_image), std::move(random_buffer),
        std::move(camera_uniform), std::move(sphere_buffer), std::move(probe_buffer),
        std::move(probe_command_buffer), std::move(probe_work_fence),
		std::move(irradiance_field_uniform),
        std::move(raytrace_command_buffer), std::move(raytrace_work_fence),
        image_descriptor_set, raytracing_descriptor_set, probe_descriptor_set,
        std::move(debug_camera_uniform), std::move(debug_vertex_buffer), debug_descriptor_set});
}

void RVPT::record_command_buffer(VK::SyncResources& current_frame, uint32_t swapchain_image_index)
{
    current_frame.command_buffer.begin();
    VkCommandBuffer cmd_buf = current_frame.command_buffer.get();

    // Image memory barrier to make sure that compute shader writes are
    // finished before sampling from the texture
    VkImageMemoryBarrier imageMemoryBarrier = {};
    imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageMemoryBarrier.image = per_frame_data[current_frame_index].output_image.image.handle;
    imageMemoryBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    imageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK::FLAGS_NONE, 0, nullptr, 0,
                         nullptr, 1, &imageMemoryBarrier);

    VkRenderPassBeginInfo rp_begin_info{};
    rp_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp_begin_info.renderPass = fullscreen_tri_render_pass;
    rp_begin_info.framebuffer = framebuffers.at(swapchain_image_index).framebuffer.handle;
    rp_begin_info.renderArea.offset = {0, 0};
    rp_begin_info.renderArea.extent = vkb_swapchain.extent;
    std::array<VkClearValue, 2> clear_values;
    clear_values[0] = {0.0f, 0.0f, 0.0f, 1.0f};
    clear_values[1] = {1.0f, 0};
    rp_begin_info.clearValueCount = static_cast<uint32_t>(clear_values.size());
    rp_begin_info.pClearValues = clear_values.data();

    vkCmdBeginRenderPass(cmd_buf, &rp_begin_info, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport{0.0f,
                        0.0f,
                        static_cast<float>(vkb_swapchain.extent.width),
                        static_cast<float>(vkb_swapchain.extent.height),
                        0.0f,
                        1.0f};
    vkCmdSetViewport(cmd_buf, 0, 1, &viewport);

    VkRect2D scissor{{0, 0}, vkb_swapchain.extent};
    vkCmdSetScissor(cmd_buf, 0, 1, &scissor);

    auto fullscreen_pipe =
        pipeline_builder.get_pipeline(rendering_resources->fullscreen_triangle_pipeline);
    vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS, fullscreen_pipe);

    vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            rendering_resources->fullscreen_triangle_pipeline_layout, 0, 1,
                            &per_frame_data[current_frame_index].image_descriptor_set.set, 0,
                            nullptr);
    vkCmdDraw(cmd_buf, 3, 1, 0, 0);

    if (show_imgui)
    {
        imgui_impl->draw(cmd_buf, current_frame_index);
    }

    vkCmdEndRenderPass(cmd_buf);
    current_frame.command_buffer.end();
}

void RVPT::record_compute_command_buffer()
{
    auto& command_buffer = per_frame_data[current_frame_index].raytrace_command_buffer;
    command_buffer.begin();
    VkCommandBuffer cmd_buf = command_buffer.get();

    uint32_t queue_family =
        compute_queue.has_value() ? compute_queue->get_family() : graphics_queue->get_family();

    VkImageMemoryBarrier probe_image_barrier = {};
    probe_image_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    probe_image_barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    probe_image_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    probe_image_barrier.image = rendering_resources->probe_texture_albedo.image.handle;
    probe_image_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    probe_image_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    probe_image_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    probe_image_barrier.dstQueueFamilyIndex = queue_family;
    probe_image_barrier.srcQueueFamilyIndex = queue_family;

    // LOOK: Here is where we dispatch probe shader
    vkCmdPipelineBarrier(cmd_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK::FLAGS_NONE, 0, nullptr, 0,
                         nullptr, 1, &probe_image_barrier);

    vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pipeline_builder.get_pipeline(rendering_resources->probe_pipeline));

    vkCmdBindDescriptorSets(
        cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, rendering_resources->probe_pipeline_layout, 0,
        1, &per_frame_data[current_frame_index].probe_descriptor_sets.set, 0, 0);

    vkCmdDispatch(cmd_buf, ceil( (float) rendering_resources->probe_texture_albedo.width / 16.0f),
                           ceil( (float) rendering_resources->probe_texture_albedo.height / 16.0f), 1);

    // Dispatch raytracing shader

    vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pipeline_builder.get_pipeline(rendering_resources->raytrace_pipeline));
    vkCmdBindDescriptorSets(
        cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, rendering_resources->raytrace_pipeline_layout, 0,
        1, &per_frame_data[current_frame_index].raytracing_descriptor_sets.set, 0, 0);

    vkCmdDispatch(cmd_buf, per_frame_data[current_frame_index].output_image.width / 16,
                  per_frame_data[current_frame_index].output_image.height / 16, 1);

    command_buffer.end();
}

void RVPT::add_sphere(Sphere sphere)
{
    spheres.emplace_back(sphere);
}

#define PI 3.1415926

void generate_samples(std::vector<glm::vec3>& output, int sqrt_num_rays)
{
    float inv_sqrt = 1.f / float(sqrt_num_rays);
    int num_rays = sqrt_num_rays * sqrt_num_rays;
    output.clear();
    output.resize(num_rays);

    int i = 0; 

    for (int y = 0; y < sqrt_num_rays; y++)
    {
        for (int x = 0; x < sqrt_num_rays; x++)
        {
            
            // First generate uniform sample
            glm::vec2 sample((x + float(rand()) / float(RAND_MAX)) * inv_sqrt,
                             (y + float(rand()) / float(RAND_MAX)) * inv_sqrt);
            
            // Then map to a sphere
            float z = 1 - (2 * sample.x);
            glm::vec3 sphere_sample(cosf(2.0f * PI * sample.y) * sqrtf(1 - (z * z)),
                                    sinf(2.0f * PI * sample.y) * sqrtf(1 - (z * z)), z);

            output[i] = sphere_sample;
            i++;
        }
    }
}

// Generate probe rays algorithmically from the
// probe positions in the irradiance field.
void RVPT::generate_probe_rays()
{
    probe_rays.clear();

    glm::ivec3 dim = ir.probe_count;
    int offset = ir.side_length;

    int num_probes = dim.x * dim.y * dim.z;

    // Generate stratified samples
    std::vector<glm::vec3> samples;
    generate_samples(samples, ir.sqrt_rays_per_probe);

    for (int p_index = 0; p_index < num_probes; p_index++)
    {

        int py = p_index / (ir.probe_count.x * ir.probe_count.z);

        int leftover = p_index - (py * ir.probe_count.x * ir.probe_count.z);
        int pz = leftover / ir.probe_count.x;
        
        int px = leftover - pz * ir.probe_count.x;

        glm::ivec3 probe_index_3d(px, py, pz);

        glm::vec3 probe_origin = probe_index_3d - ((dim - glm::ivec3(1)) / 2);

        probe_origin *= offset;
        
        probe_origin += ir.field_origin;

        int x = 0, y = 0;

        for (int i = 0; i < samples.size(); i++)
        {
            probe_rays.emplace_back(ProbeRay(probe_origin,
                                             glm::normalize(samples[i]),
                                             p_index,
                                             glm::vec2(x, y)));

            x++;
            if (x >= ir.sqrt_rays_per_probe)
            {
                x = 0;
                y++;
            }
        }
    }
    
    need_generate_probe_rays = false;
}

void RVPT::get_asset_path(std::string& asset_path)
{
    // I'm 99% sure this method can be made a lot faster, which it 100% can.
    // but I really cannot be asked.
    // Is this method even required? Could we be doing something else? Most likely yes.
    // Any idea dm me on discord Legend#4321

    if (source_folder == "")
    {
        fmt::print("source folder not set, unable to get asset path\n");
        return;
    }

    asset_path = source_folder + "/assets/" + asset_path;
}
