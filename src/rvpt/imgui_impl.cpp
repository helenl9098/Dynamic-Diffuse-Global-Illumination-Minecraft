#include "imgui_impl.h"

#include <stdio.h>
#include <array>

#include <imgui.h>

/*
#version 450 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
layout(location = 2) in vec4 aColor;
layout(push_constant) uniform uPushConstant { vec2 uScale; vec2 uTranslate; } pc;
out gl_PerVertex { vec4 gl_Position; };
layout(location = 0) out struct { vec4 Color; vec2 UV; } Out;
void main()
{
    Out.Color = aColor;
    Out.UV = aUV;
    gl_Position = vec4(aPos * pc.uScale + pc.uTranslate, 0, 1);
}
*/
static uint32_t glsl_shader_vert_spv[] = {
    0x07230203, 0x00010000, 0x00080001, 0x0000002e, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x000a000f, 0x00000000, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000b, 0x0000000f, 0x00000015,
    0x0000001b, 0x0000001c, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d,
    0x00000000, 0x00030005, 0x00000009, 0x00000000, 0x00050006, 0x00000009, 0x00000000, 0x6f6c6f43,
    0x00000072, 0x00040006, 0x00000009, 0x00000001, 0x00005655, 0x00030005, 0x0000000b, 0x0074754f,
    0x00040005, 0x0000000f, 0x6c6f4361, 0x0000726f, 0x00030005, 0x00000015, 0x00565561, 0x00060005,
    0x00000019, 0x505f6c67, 0x65567265, 0x78657472, 0x00000000, 0x00060006, 0x00000019, 0x00000000,
    0x505f6c67, 0x7469736f, 0x006e6f69, 0x00030005, 0x0000001b, 0x00000000, 0x00040005, 0x0000001c,
    0x736f5061, 0x00000000, 0x00060005, 0x0000001e, 0x73755075, 0x6e6f4368, 0x6e617473, 0x00000074,
    0x00050006, 0x0000001e, 0x00000000, 0x61635375, 0x0000656c, 0x00060006, 0x0000001e, 0x00000001,
    0x61725475, 0x616c736e, 0x00006574, 0x00030005, 0x00000020, 0x00006370, 0x00040047, 0x0000000b,
    0x0000001e, 0x00000000, 0x00040047, 0x0000000f, 0x0000001e, 0x00000002, 0x00040047, 0x00000015,
    0x0000001e, 0x00000001, 0x00050048, 0x00000019, 0x00000000, 0x0000000b, 0x00000000, 0x00030047,
    0x00000019, 0x00000002, 0x00040047, 0x0000001c, 0x0000001e, 0x00000000, 0x00050048, 0x0000001e,
    0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x0000001e, 0x00000001, 0x00000023, 0x00000008,
    0x00030047, 0x0000001e, 0x00000002, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002,
    0x00030016, 0x00000006, 0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040017,
    0x00000008, 0x00000006, 0x00000002, 0x0004001e, 0x00000009, 0x00000007, 0x00000008, 0x00040020,
    0x0000000a, 0x00000003, 0x00000009, 0x0004003b, 0x0000000a, 0x0000000b, 0x00000003, 0x00040015,
    0x0000000c, 0x00000020, 0x00000001, 0x0004002b, 0x0000000c, 0x0000000d, 0x00000000, 0x00040020,
    0x0000000e, 0x00000001, 0x00000007, 0x0004003b, 0x0000000e, 0x0000000f, 0x00000001, 0x00040020,
    0x00000011, 0x00000003, 0x00000007, 0x0004002b, 0x0000000c, 0x00000013, 0x00000001, 0x00040020,
    0x00000014, 0x00000001, 0x00000008, 0x0004003b, 0x00000014, 0x00000015, 0x00000001, 0x00040020,
    0x00000017, 0x00000003, 0x00000008, 0x0003001e, 0x00000019, 0x00000007, 0x00040020, 0x0000001a,
    0x00000003, 0x00000019, 0x0004003b, 0x0000001a, 0x0000001b, 0x00000003, 0x0004003b, 0x00000014,
    0x0000001c, 0x00000001, 0x0004001e, 0x0000001e, 0x00000008, 0x00000008, 0x00040020, 0x0000001f,
    0x00000009, 0x0000001e, 0x0004003b, 0x0000001f, 0x00000020, 0x00000009, 0x00040020, 0x00000021,
    0x00000009, 0x00000008, 0x0004002b, 0x00000006, 0x00000028, 0x00000000, 0x0004002b, 0x00000006,
    0x00000029, 0x3f800000, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8,
    0x00000005, 0x0004003d, 0x00000007, 0x00000010, 0x0000000f, 0x00050041, 0x00000011, 0x00000012,
    0x0000000b, 0x0000000d, 0x0003003e, 0x00000012, 0x00000010, 0x0004003d, 0x00000008, 0x00000016,
    0x00000015, 0x00050041, 0x00000017, 0x00000018, 0x0000000b, 0x00000013, 0x0003003e, 0x00000018,
    0x00000016, 0x0004003d, 0x00000008, 0x0000001d, 0x0000001c, 0x00050041, 0x00000021, 0x00000022,
    0x00000020, 0x0000000d, 0x0004003d, 0x00000008, 0x00000023, 0x00000022, 0x00050085, 0x00000008,
    0x00000024, 0x0000001d, 0x00000023, 0x00050041, 0x00000021, 0x00000025, 0x00000020, 0x00000013,
    0x0004003d, 0x00000008, 0x00000026, 0x00000025, 0x00050081, 0x00000008, 0x00000027, 0x00000024,
    0x00000026, 0x00050051, 0x00000006, 0x0000002a, 0x00000027, 0x00000000, 0x00050051, 0x00000006,
    0x0000002b, 0x00000027, 0x00000001, 0x00070050, 0x00000007, 0x0000002c, 0x0000002a, 0x0000002b,
    0x00000028, 0x00000029, 0x00050041, 0x00000011, 0x0000002d, 0x0000001b, 0x0000000d, 0x0003003e,
    0x0000002d, 0x0000002c, 0x000100fd, 0x00010038};

/*
#version 450 core
layout(location = 0) out vec4 fColor;
layout(set=0, binding=0) uniform sampler2D sTexture;
layout(location = 0) in struct { vec4 Color; vec2 UV; } In;
void main()
{
    fColor = In.Color * texture(sTexture, In.UV.st);
}
*/
static uint32_t glsl_shader_frag_spv[] = {
    0x07230203, 0x00010000, 0x00080001, 0x0000001e, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0007000f, 0x00000004, 0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x0000000d, 0x00030010,
    0x00000004, 0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004, 0x6e69616d,
    0x00000000, 0x00040005, 0x00000009, 0x6c6f4366, 0x0000726f, 0x00030005, 0x0000000b, 0x00000000,
    0x00050006, 0x0000000b, 0x00000000, 0x6f6c6f43, 0x00000072, 0x00040006, 0x0000000b, 0x00000001,
    0x00005655, 0x00030005, 0x0000000d, 0x00006e49, 0x00050005, 0x00000016, 0x78655473, 0x65727574,
    0x00000000, 0x00040047, 0x00000009, 0x0000001e, 0x00000000, 0x00040047, 0x0000000d, 0x0000001e,
    0x00000000, 0x00040047, 0x00000016, 0x00000022, 0x00000000, 0x00040047, 0x00000016, 0x00000021,
    0x00000000, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006,
    0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040020, 0x00000008, 0x00000003,
    0x00000007, 0x0004003b, 0x00000008, 0x00000009, 0x00000003, 0x00040017, 0x0000000a, 0x00000006,
    0x00000002, 0x0004001e, 0x0000000b, 0x00000007, 0x0000000a, 0x00040020, 0x0000000c, 0x00000001,
    0x0000000b, 0x0004003b, 0x0000000c, 0x0000000d, 0x00000001, 0x00040015, 0x0000000e, 0x00000020,
    0x00000001, 0x0004002b, 0x0000000e, 0x0000000f, 0x00000000, 0x00040020, 0x00000010, 0x00000001,
    0x00000007, 0x00090019, 0x00000013, 0x00000006, 0x00000001, 0x00000000, 0x00000000, 0x00000000,
    0x00000001, 0x00000000, 0x0003001b, 0x00000014, 0x00000013, 0x00040020, 0x00000015, 0x00000000,
    0x00000014, 0x0004003b, 0x00000015, 0x00000016, 0x00000000, 0x0004002b, 0x0000000e, 0x00000018,
    0x00000001, 0x00040020, 0x00000019, 0x00000001, 0x0000000a, 0x00050036, 0x00000002, 0x00000004,
    0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x00050041, 0x00000010, 0x00000011, 0x0000000d,
    0x0000000f, 0x0004003d, 0x00000007, 0x00000012, 0x00000011, 0x0004003d, 0x00000014, 0x00000017,
    0x00000016, 0x00050041, 0x00000019, 0x0000001a, 0x0000000d, 0x00000018, 0x0004003d, 0x0000000a,
    0x0000001b, 0x0000001a, 0x00050057, 0x00000007, 0x0000001c, 0x00000017, 0x0000001b, 0x00050085,
    0x00000007, 0x0000001d, 0x00000012, 0x0000001c, 0x0003003e, 0x00000009, 0x0000001d, 0x000100fd,
    0x00010038};

auto create_font_texture(VkDevice device, VK::MemoryAllocator& memory_allocator,
                         VK::Queue& graphics_queue)
{
    ImGuiIO& io = ImGui::GetIO();

    unsigned char* pixels;
    int width, height;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    size_t upload_size = width * height * 4 * sizeof(char);

    VK::Image font_image(device, memory_allocator, graphics_queue, "imgui_font_image",
                         VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, width, height,
                         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT,
                         upload_size, VK::MemoryUsage::gpu);

    VK::Buffer upload_buffer(device, memory_allocator, "imgui_font_upload_buffer",
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT, upload_size,
                             VK::MemoryUsage::cpu_to_gpu);

    upload_buffer.copy_bytes(pixels, upload_size);
    upload_buffer.flush();

    VK::CommandBuffer command_buffer(device, graphics_queue, "imgui_font_upload_command_buffer");
    command_buffer.begin();

    VkImageMemoryBarrier copy_barrier[1] = {};
    copy_barrier[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    copy_barrier[0].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    copy_barrier[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    copy_barrier[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    copy_barrier[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    copy_barrier[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    copy_barrier[0].image = font_image.get();
    copy_barrier[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy_barrier[0].subresourceRange.levelCount = 1;
    copy_barrier[0].subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(command_buffer.get(), VK_PIPELINE_STAGE_HOST_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, copy_barrier);
    VkBufferImageCopy region = {};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent.width = width;
    region.imageExtent.height = height;
    region.imageExtent.depth = 1;
    vkCmdCopyBufferToImage(command_buffer.get(), upload_buffer.get(), font_image.get(),
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    VkImageMemoryBarrier use_barrier[1] = {};
    use_barrier[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    use_barrier[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    use_barrier[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    use_barrier[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    use_barrier[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    use_barrier[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    use_barrier[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    use_barrier[0].image = font_image.get();
    use_barrier[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    use_barrier[0].subresourceRange.levelCount = 1;
    use_barrier[0].subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(command_buffer.get(), VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0, NULL, 1,
                         use_barrier);
    command_buffer.end();
    VK::Fence done_fence(device, "imgui_font_upload_fence");
    graphics_queue.submit(command_buffer, done_fence);
    done_fence.wait();
    font_image.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    return std::move(font_image);
}

auto create_descriptor_pool(VkDevice device, VkSampler font_sampler)
{
    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT,
         &font_sampler}};
    return VK::DescriptorPool{device, bindings, 1, "imgui_descriptor_pool"};
}

auto create_pipeline_layout(VkDevice device, VK::PipelineBuilder& pipeline_builder,
                            VkDescriptorSetLayout layout)
{
    std::vector<VkDescriptorSetLayout> layouts = {layout};
    std::vector<VkPushConstantRange> push_constants = {
        {VK_SHADER_STAGE_VERTEX_BIT, sizeof(float) * 0, sizeof(float) * 4}};
    return pipeline_builder.create_layout(layouts, push_constants, "imgui_pipeline_layout");
}

auto create_vert_shader(VkDevice device)
{
    std::vector<uint32_t> vert_code;
    vert_code.resize(324);  // length of glsl_shader_vert_spv
    for (int i = 0; i < 324; i++) vert_code[i] = glsl_shader_vert_spv[i];

    return vert_code;
}

auto create_frag_shader(VkDevice device)
{
    std::vector<uint32_t> frag_code;
    frag_code.resize(193);  // length of glsl_shader_frag_spv
    for (int i = 0; i < 193; i++) frag_code[i] = glsl_shader_frag_spv[i];

    return device, frag_code;
}

auto create_pipeline(VkDevice device, VK::PipelineBuilder& pipeline_builder,
                     VkPipelineLayout layout, VkRenderPass render_pass, VkExtent2D extent)
{
    auto vert = create_vert_shader(device);
    auto frag = create_frag_shader(device);

    std::vector<VkVertexInputBindingDescription> binding_desc = {
        {0, sizeof(ImDrawVert), VK_VERTEX_INPUT_RATE_VERTEX}};

    std::vector<VkVertexInputAttributeDescription> attribute_desc = {
        {0, binding_desc[0].binding, VK_FORMAT_R32G32_SFLOAT, IM_OFFSETOF(ImDrawVert, pos)},
        {1, binding_desc[0].binding, VK_FORMAT_R32G32_SFLOAT, IM_OFFSETOF(ImDrawVert, uv)},
        {2, binding_desc[0].binding, VK_FORMAT_R8G8B8A8_UNORM, IM_OFFSETOF(ImDrawVert, col)}};

    VK::GraphicsPipelineDetails pipe_details;
    pipe_details.name = "imgui_pipeline";
    pipe_details.pipeline_layout = layout;
    pipe_details.spirv_vert_data = vert;
    pipe_details.spirv_frag_data = frag;
    pipe_details.binding_desc = binding_desc;
    pipe_details.attribute_desc = attribute_desc;
    pipe_details.enable_blending = true;
    pipe_details.render_pass = render_pass;
    pipe_details.extent = extent;
    auto pipeline = pipeline_builder.create_immutable_pipeline(pipe_details);

    return VK::HandleWrapper<VkPipeline, PFN_vkDestroyPipeline>(device, pipeline,
                                                                vkDestroyPipeline);
}

ImguiImpl::ImguiImpl(VkDevice device, VK::Queue& graphics_queue,
                     VK::PipelineBuilder& pipeline_builder, VK::MemoryAllocator& memory_allocator,
                     VkRenderPass render_pass, VkExtent2D extent, uint32_t max_frames_in_flight)
    : device(device),
      memory_allocator(memory_allocator),
      font_image(create_font_texture(device, memory_allocator, graphics_queue)),
      pool(create_descriptor_pool(device, font_image.sampler.handle)),
      descriptor_set(pool.allocate("imgui_descriptor_set")),
      pipeline_layout(create_pipeline_layout(device, pipeline_builder, pool.layout())),
      pipeline(create_pipeline(device, pipeline_builder, pipeline_layout, render_pass, extent))
{
    for (uint32_t i = 0; i < max_frames_in_flight; i++)
    {
        vertex_buffers.emplace_back(
            device, memory_allocator, "imgui_vertex_buffer_" + std::to_string(i),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, 1000 * 36, VK::MemoryUsage::cpu_to_gpu);
        index_buffers.emplace_back(
            device, memory_allocator, "imgui_index_buffer_" + std::to_string(i),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT, 1000 * 4, VK::MemoryUsage::cpu_to_gpu);
    }

    VkDescriptorImageInfo write_descriptor[] = {font_image.descriptor_info()};
    VkWriteDescriptorSet write_desc[1] = {};
    write_desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write_desc[0].dstSet = descriptor_set.set;
    write_desc[0].descriptorCount = 1;
    write_desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write_desc[0].pImageInfo = write_descriptor;
    vkUpdateDescriptorSets(device, 1, write_desc, 0, nullptr);

    ImGuiIO& io = ImGui::GetIO();
    io.BackendRendererName = "imgui_impl_vulkan";
    io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;  // allow for large meshes.
}

void ImguiImpl::draw(VkCommandBuffer command_buffer, uint32_t frame_index)
{
    ImGui::Render();
    ImDrawData* draw_data = ImGui::GetDrawData();

    // Avoid rendering when minimized, scale coordinates for retina displays (screen coordinates !=
    // framebuffer coordinates)
    int fb_width = (int)(draw_data->DisplaySize.x * draw_data->FramebufferScale.x);
    int fb_height = (int)(draw_data->DisplaySize.y * draw_data->FramebufferScale.y);
    if (fb_width <= 0 || fb_height <= 0) return;

    if (draw_data->TotalVtxCount > 0)
    {
        // Create or resize the vertex/index buffers
        size_t vertex_size = draw_data->TotalVtxCount * sizeof(ImDrawVert);
        size_t index_size = draw_data->TotalIdxCount * sizeof(ImDrawIdx);

        std::vector<ImDrawVert> vertex_data;
        vertex_data.reserve(draw_data->TotalVtxCount);
        std::vector<ImDrawIdx> index_data;
        index_data.reserve(draw_data->TotalIdxCount);

        for (int n = 0; n < draw_data->CmdListsCount; n++)
        {
            const ImDrawList* cmd_list = draw_data->CmdLists[n];
            vertex_data.insert(vertex_data.end(), cmd_list->VtxBuffer.begin(),
                               cmd_list->VtxBuffer.end());
            index_data.insert(index_data.end(), cmd_list->IdxBuffer.begin(),
                              cmd_list->IdxBuffer.end());
        }

        if (vertex_buffers.at(frame_index).size() < vertex_size)
        {
            vertex_buffers.at(frame_index) = VK::Buffer(device, memory_allocator, "imgui_vertices",
                                                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                                        vertex_size, VK::MemoryUsage::cpu_to_gpu);
        }
        if (index_buffers.at(frame_index).size() < index_size)
        {
            index_buffers.at(frame_index) = VK::Buffer(device, memory_allocator, "imgui_indices",
                                                       VK_BUFFER_USAGE_INDEX_BUFFER_BIT, index_size,
                                                       VK::MemoryUsage::cpu_to_gpu);
        }

        vertex_buffers.at(frame_index).copy_to(vertex_data);
        index_buffers.at(frame_index).copy_to(index_data);
        vertex_buffers.at(frame_index).flush();
        index_buffers.at(frame_index).flush();
    }

    // Bind pipeline and descriptor sets:
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.handle);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1,
                            &descriptor_set.set, 0, NULL);

    if (draw_data->TotalVtxCount > 0)
    {
        VkBuffer vert_buffers = vertex_buffers.at(frame_index).get();
        VkDeviceSize vertex_offset = 0;
        vkCmdBindVertexBuffers(command_buffer, 0, 1, &vert_buffers, &vertex_offset);
        vkCmdBindIndexBuffer(command_buffer, index_buffers.at(frame_index).get(), 0,
                             sizeof(ImDrawIdx) == 2 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
    }

    VkViewport viewport{0,   0,  static_cast<float>(fb_width), static_cast<float>(fb_height),
                        0.f, 1.f};
    vkCmdSetViewport(command_buffer, 0, 1, &viewport);

    // Setup scale and translation:
    // Our visible imgui space lies from draw_data->DisplayPps (top left) to
    // draw_data->DisplayPos+data_data->DisplaySize (bottom right). DisplayPos is (0,0) for single
    // viewport apps.

    std::array<float, 2> scale = {2.0f / draw_data->DisplaySize.x, 2.0f / draw_data->DisplaySize.y};
    std::array<float, 2> translate = {-1.0f - draw_data->DisplayPos.x * scale[0],
                                      -1.0f - draw_data->DisplayPos.y * scale[1]};
    vkCmdPushConstants(command_buffer, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 8,
                       scale.data());
    vkCmdPushConstants(command_buffer, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 8, 8,
                       translate.data());

    // Will project scissor/clipping rectangles into framebuffer space
    ImVec2 clip_off = draw_data->DisplayPos;  // (0,0) unless using multi-viewports
    ImVec2 clip_scale =
        draw_data->FramebufferScale;  // (1,1) unless using retina display which are often (2,2)

    // Render command lists
    // (Because we merged all buffers into a single one, we maintain our own offset into them)
    int global_vtx_offset = 0;
    int global_idx_offset = 0;
    for (int n = 0; n < draw_data->CmdListsCount; n++)
    {
        const ImDrawList* cmd_list = draw_data->CmdLists[n];
        for (int cmd_i = 0; cmd_i < cmd_list->CmdBuffer.Size; cmd_i++)
        {
            const ImDrawCmd* p_cmd = &cmd_list->CmdBuffer[cmd_i];
            if (p_cmd->UserCallback != NULL)
            {
                // // User callback, registered via ImDrawList::AddCallback()
                // // (ImDrawCallback_ResetRenderState is a special callback value used by the user
                // to
                // // request the renderer to reset render state.)
                // if (p_cmd->UserCallback == ImDrawCallback_ResetRenderState)
                //     ImGui_ImplVulkan_SetupRenderState(draw_data, command_buffer, rb, fb_width,
                //                                       fb_height);
                // else
                //     p_cmd->UserCallback(cmd_list, p_cmd);
            }
            else
            {
                // Project scissor/clipping rectangles into framebuffer space
                ImVec4 clip_rect;
                clip_rect.x = (p_cmd->ClipRect.x - clip_off.x) * clip_scale.x;
                clip_rect.y = (p_cmd->ClipRect.y - clip_off.y) * clip_scale.y;
                clip_rect.z = (p_cmd->ClipRect.z - clip_off.x) * clip_scale.x;
                clip_rect.w = (p_cmd->ClipRect.w - clip_off.y) * clip_scale.y;

                if (clip_rect.x < fb_width && clip_rect.y < fb_height && clip_rect.z >= 0.0f &&
                    clip_rect.w >= 0.0f)
                {
                    // Negative offsets are illegal for vkCmdSetScissor
                    if (clip_rect.x < 0.0f) clip_rect.x = 0.0f;
                    if (clip_rect.y < 0.0f) clip_rect.y = 0.0f;

                    // Apply scissor/clipping rectangle
                    VkRect2D scissor;
                    scissor.offset.x = static_cast<int32_t>(clip_rect.x);
                    scissor.offset.y = static_cast<int32_t>(clip_rect.y);
                    scissor.extent.width = static_cast<uint32_t>(clip_rect.z - clip_rect.x);
                    scissor.extent.height = static_cast<uint32_t>(clip_rect.w - clip_rect.y);
                    vkCmdSetScissor(command_buffer, 0, 1, &scissor);

                    // Draw
                    vkCmdDrawIndexed(command_buffer, p_cmd->ElemCount, 1,
                                     p_cmd->IdxOffset + global_idx_offset,
                                     p_cmd->VtxOffset + global_vtx_offset, 0);
                }
            }
        }
        global_idx_offset += cmd_list->IdxBuffer.Size;
        global_vtx_offset += cmd_list->VtxBuffer.Size;
    }
}
