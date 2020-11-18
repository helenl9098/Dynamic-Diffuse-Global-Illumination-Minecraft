#pragma once

#include "vk_util.h"

class ImguiImpl
{
public:
    ImguiImpl(VkDevice device, VK::Queue& graphics_queue, VK::PipelineBuilder& pipeline_builder,
              VK::MemoryAllocator& memory_allocator, VkRenderPass render_pass, VkExtent2D extent,
              uint32_t max_frames_in_flight);

    void new_frame();
    void draw(VkCommandBuffer command_buffer, uint32_t frame_index);

private:
    VkDevice device;
    VK::MemoryAllocator& memory_allocator;
    VK::Image font_image;
    VK::DescriptorPool pool;
    VK::DescriptorSet descriptor_set;
    VkPipelineLayout pipeline_layout;
    VK::HandleWrapper<VkPipeline, PFN_vkDestroyPipeline> pipeline;

    std::vector<VK::Buffer> vertex_buffers;
    std::vector<VK::Buffer> index_buffers;
};
