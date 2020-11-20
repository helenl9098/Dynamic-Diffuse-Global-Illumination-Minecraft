#include "vk_util.h"

#include <cstring>

#include <array>
#include <algorithm>
#include <fstream>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <utility>

const char* error_str(const VkResult result)
{
    switch (result)
    {
#define STR(r)   \
    case VK_##r: \
        return #r
        STR(NOT_READY);
        STR(TIMEOUT);
        STR(EVENT_SET);
        STR(EVENT_RESET);
        STR(INCOMPLETE);
        STR(ERROR_OUT_OF_HOST_MEMORY);
        STR(ERROR_OUT_OF_DEVICE_MEMORY);
        STR(ERROR_INITIALIZATION_FAILED);
        STR(ERROR_DEVICE_LOST);
        STR(ERROR_MEMORY_MAP_FAILED);
        STR(ERROR_LAYER_NOT_PRESENT);
        STR(ERROR_EXTENSION_NOT_PRESENT);
        STR(ERROR_FEATURE_NOT_PRESENT);
        STR(ERROR_INCOMPATIBLE_DRIVER);
        STR(ERROR_TOO_MANY_OBJECTS);
        STR(ERROR_FORMAT_NOT_SUPPORTED);
        STR(ERROR_SURFACE_LOST_KHR);
        STR(ERROR_NATIVE_WINDOW_IN_USE_KHR);
        STR(SUBOPTIMAL_KHR);
        STR(ERROR_OUT_OF_DATE_KHR);
        STR(ERROR_INCOMPATIBLE_DISPLAY_KHR);
        STR(ERROR_VALIDATION_FAILED_EXT);
        STR(ERROR_INVALID_SHADER_NV);
#undef STR
        default:
            return "UNKNOWN_ERROR";
    }
}
namespace VK
{
template <typename T>
auto gather_vk_types(std::vector<T> const& values)
{
    std::vector<decltype(std::declval<T>().get())> types;
    for (auto const& val : values) types.push_back(val.get());
    return types;
}

// Debug Utils Helper
DebugUtilHelper debug_utils_helper;

void setup_debug_util_helper(VkDevice device) { debug_utils_helper = DebugUtilHelper(device); }

// Fence

constexpr long DEFAULT_FENCE_TIMEOUT = 1000000000;

auto create_fence(VkDevice device, VkFenceCreateFlags flags)
{
    VkFenceCreateInfo fenceCreateInfo{};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = flags;
    VkFence handle;
    VK_CHECK_RESULT(vkCreateFence(device, &fenceCreateInfo, nullptr, &handle));
    return HandleWrapper(device, handle, vkDestroyFence);
}

Fence::Fence(VkDevice device, std::string const& name, VkFenceCreateFlags flags)
    : fence(create_fence(device, flags))
{
    debug_utils_helper.set_debug_object_name(VK_OBJECT_TYPE_FENCE, fence.handle, name);
}

bool Fence::check() const
{
    VkResult out = vkGetFenceStatus(fence.device, fence.handle);
    if (out == VK_SUCCESS)
        return true;
    else if (out == VK_NOT_READY)
        return false;
    assert(out == VK_SUCCESS || out == VK_NOT_READY);
    return false;
}

void Fence::wait(bool condition) const
{
    vkWaitForFences(fence.device, 1, &fence.handle, condition, DEFAULT_FENCE_TIMEOUT);
}

VkFence Fence::get() const { return fence.handle; }

void Fence::reset() const { vkResetFences(fence.device, 1, &fence.handle); }

// Semaphore

auto create_semaphore(VkDevice device)
{
    VkSemaphoreCreateInfo semaphoreCreateInfo{};
    semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkSemaphore semaphore;
    VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &semaphore));
    return HandleWrapper(device, semaphore, vkDestroySemaphore);
}

Semaphore::Semaphore(VkDevice device, std::string const& name) : semaphore(create_semaphore(device))
{
    debug_utils_helper.set_debug_object_name(VK_OBJECT_TYPE_SEMAPHORE, semaphore.handle, name);
}

VkSemaphore Semaphore::get() const { return semaphore.handle; }

// Queue

Queue::Queue(VkDevice device, uint32_t queue_family, std::string const& name, uint32_t queue_index)
    : queue_family(queue_family)
{
    vkGetDeviceQueue(device, queue_family, queue_index, &queue);
    debug_utils_helper.set_debug_object_name(VK_OBJECT_TYPE_QUEUE, queue, name);
}

void Queue::submit(CommandBuffer const& command_buffer, Fence& fence)
{
    // auto cmd_bufs = gather_vk_types({command_buffer});
    VkCommandBuffer cmd_buf = command_buffer.get();

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd_buf;

    submit(submit_info, fence);
}

void Queue::submit(CommandBuffer const& command_buffer, Fence const& fence,
                   Semaphore const& wait_semaphore, Semaphore const& signal_semaphore,
                   VkPipelineStageFlags const stage_mask)
{
    VkCommandBuffer cmd_buf = command_buffer.get();
    VkSemaphore signal_sem = signal_semaphore.get();
    VkSemaphore wait_sem = wait_semaphore.get();

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd_buf;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &signal_sem;
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = &wait_sem;
    submit_info.pWaitDstStageMask = &stage_mask;

    submit(submit_info, fence);
}

void Queue::submit(VkSubmitInfo const& submitInfo, Fence const& fence)
{
    std::lock_guard lock(submit_mutex);
    VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, fence.get()));
}

int Queue::get_family() const { return queue_family; }
VkQueue Queue::get() const { return queue; }

VkResult Queue::presentation_submit(VkPresentInfoKHR present_info)
{
    std::lock_guard lock(submit_mutex);
    return vkQueuePresentKHR(queue, &present_info);
}

void Queue::wait_idle()
{
    std::lock_guard lock(submit_mutex);
    vkQueueWaitIdle(queue);
}

// Command Pool

auto create_command_pool(VkDevice device, Queue const& queue, VkCommandPoolCreateFlags flags)
{
    VkCommandPoolCreateInfo cmd_pool_info{};
    cmd_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmd_pool_info.queueFamilyIndex = queue.get_family();
    cmd_pool_info.flags = flags;
    VkCommandPool pool;
    VK_CHECK_RESULT(vkCreateCommandPool(device, &cmd_pool_info, nullptr, &pool));
    return HandleWrapper(device, pool, vkDestroyCommandPool);
}

CommandPool::CommandPool(VkDevice device, Queue const& queue, std::string const& name,
                         VkCommandPoolCreateFlags flags)
    : pool(create_command_pool(device, queue, flags))
{
    debug_utils_helper.set_debug_object_name(VK_OBJECT_TYPE_COMMAND_POOL, pool.handle, name);
}

VkCommandBuffer CommandPool::allocate()
{
    VkCommandBuffer command_buffer;

    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = pool.handle;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    VK_CHECK_RESULT(vkAllocateCommandBuffers(pool.device, &alloc_info, &command_buffer));
    return command_buffer;
}
void CommandPool::free(VkCommandBuffer command_buffer)
{
    vkFreeCommandBuffers(pool.device, pool.handle, 1, &command_buffer);
}

// Command Buffer
CommandBuffer::CommandBuffer(VkDevice device, Queue const& queue, std::string const& name)
    : device(device),
      queue(&queue),
      pool(device, queue, name + "'s command pool", VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)
{
    command_buffer = pool.allocate();
    debug_utils_helper.set_debug_object_name(VK_OBJECT_TYPE_COMMAND_BUFFER, command_buffer, name);
}
CommandBuffer::~CommandBuffer()
{
    if (command_buffer != nullptr) pool.free(command_buffer);
}

CommandBuffer::CommandBuffer(CommandBuffer&& other) noexcept
    : device(other.device),
      queue(other.queue),
      pool(std::move(other.pool)),
      command_buffer(other.command_buffer)
{
    other.command_buffer = nullptr;
}
CommandBuffer& CommandBuffer::operator=(CommandBuffer&& other) noexcept
{
    if (this != &other)
    {
        if (command_buffer != nullptr) pool.free(command_buffer);
        device = other.device;
        queue = other.queue;
        pool = std::move(other.pool);
        command_buffer = other.command_buffer;
        other.command_buffer = VK_NULL_HANDLE;
    }
    return *this;
}

VkCommandBuffer CommandBuffer::get() const { return command_buffer; }

void CommandBuffer::begin(VkCommandBufferUsageFlags flags)
{
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = flags;

    VK_CHECK_RESULT(vkBeginCommandBuffer(command_buffer, &beginInfo));
}
void CommandBuffer::end() { VK_CHECK_RESULT(vkEndCommandBuffer(command_buffer)); }
void CommandBuffer::reset() { VK_CHECK_RESULT(vkResetCommandBuffer(command_buffer, {})); }

// Frame Resources
SyncResources::SyncResources(VkDevice device, Queue& graphics_queue, Queue& present_queue,
                             VkSwapchainKHR swapchain)
    : graphics_queue(graphics_queue),
      present_queue(present_queue),
      swapchain(swapchain),
      image_avail_sem(device, "image_avail_sem"),
      render_finish_sem(device, "render_finish_sem"),
      command_fence(device, "sync_res_fence", VK_FENCE_CREATE_SIGNALED_BIT),
      command_buffer(device, graphics_queue, "sync_res_command_buffer")
{
}

void SyncResources::submit()
{
    graphics_queue.submit(command_buffer, command_fence, image_avail_sem, render_finish_sem,
                          VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
}
VkResult SyncResources::present(uint32_t image_index)
{
    VkSemaphore wait_sem = render_finish_sem.get();
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &wait_sem;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapchain;
    presentInfo.pImageIndices = &image_index;

    return present_queue.presentation_submit(presentInfo);
}

// DescriptorUse
DescriptorUse::DescriptorUse(uint32_t bind_point, uint32_t count, VkDescriptorType type,
                             DescriptorUseVector descriptor_use_data)
    : bind_point(bind_point), count(count), type(type), descriptor_use_data(descriptor_use_data)
{
}

VkWriteDescriptorSet DescriptorUse::get_write_descriptor_set(VkDescriptorSet set)
{
    VkWriteDescriptorSet writeDescriptorSet{};
    writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet.dstSet = set;
    writeDescriptorSet.descriptorType = type;
    writeDescriptorSet.dstBinding = bind_point;
    writeDescriptorSet.descriptorCount = count;

    if (descriptor_use_data.index() == 0)
        writeDescriptorSet.pBufferInfo =
            std::get<std::vector<VkDescriptorBufferInfo>>(descriptor_use_data).data();
    else if (descriptor_use_data.index() == 1)
        writeDescriptorSet.pImageInfo =
            std::get<std::vector<VkDescriptorImageInfo>>(descriptor_use_data).data();
    else if (descriptor_use_data.index() == 2)
        writeDescriptorSet.pTexelBufferView =
            std::get<std::vector<VkBufferView>>(descriptor_use_data).data();
    return writeDescriptorSet;
}

// DescriptorSet

DescriptorSet::DescriptorSet(VkDevice device, VkDescriptorSet set, VkDescriptorSetLayout layout,
                             std::string const& name)
    : device(device), set(set), layout(layout)
{
    debug_utils_helper.set_debug_object_name(VK_OBJECT_TYPE_DESCRIPTOR_SET, set, name);
}
void DescriptorSet::update(std::vector<DescriptorUse> descriptors) const
{
    std::vector<VkWriteDescriptorSet> writes;
    for (auto& descriptor : descriptors)
    {
        writes.push_back(descriptor.get_write_descriptor_set(set));
    }

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}
void DescriptorSet::bind(VkCommandBuffer cmdBuf, VkPipelineBindPoint bind_point,
                         VkPipelineLayout layout, uint32_t location) const
{
    vkCmdBindDescriptorSets(cmdBuf, bind_point, layout, location, 1, &set, 0, nullptr);
}

// DescriptorPool

auto create_descriptor_set_layout(VkDevice device,
                                  std::vector<VkDescriptorSetLayoutBinding> const& bindings)
{
    VkDescriptorSetLayoutCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    create_info.bindingCount = static_cast<uint32_t>(bindings.size());
    create_info.pBindings = bindings.data();

    VkDescriptorSetLayout layout;
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &create_info, nullptr, &layout));
    return HandleWrapper(device, layout, vkDestroyDescriptorSetLayout);
}
auto create_descriptor_pool(VkDevice device,
                            std::vector<VkDescriptorSetLayoutBinding> const& bindings,
                            uint32_t max_sets)
{
    std::unordered_map<VkDescriptorType, uint32_t> descriptor_map;
    for (auto& binding : bindings)
    {
        descriptor_map[binding.descriptorType] += binding.descriptorCount;
    }
    std::vector<VkDescriptorPoolSize> pool_sizes;
    for (auto& [type, count] : descriptor_map)
    {
        pool_sizes.push_back(VkDescriptorPoolSize{type, max_sets * count});
    }

    VkDescriptorPoolCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    create_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    create_info.pPoolSizes = pool_sizes.data();
    create_info.maxSets = max_sets;

    VkDescriptorPool pool;
    VK_CHECK_RESULT(vkCreateDescriptorPool(device, &create_info, nullptr, &pool));
    return HandleWrapper(device, pool, vkDestroyDescriptorPool);
}

DescriptorPool::DescriptorPool(VkDevice device,
                               std::vector<VkDescriptorSetLayoutBinding> const& bindings,
                               uint32_t count, std::string const& name)
    : vk_layout(create_descriptor_set_layout(device, bindings)),
      pool(create_descriptor_pool(device, bindings, count)),
      bindings(bindings),
      max_sets(count)
{
    debug_utils_helper.set_debug_object_name(VK_OBJECT_TYPE_DESCRIPTOR_POOL, pool.handle, name);
    debug_utils_helper.set_debug_object_name(VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, vk_layout.handle,
                                             name + "_layout");
}

DescriptorSet DescriptorPool::allocate(std::string const& name)
{
    assert(current_sets < max_sets);
    VkDescriptorSet set;
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = pool.handle;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &vk_layout.handle;

    VkResult res = vkAllocateDescriptorSets(pool.device, &alloc_info, &set);
    assert(res == VK_SUCCESS);

    return DescriptorSet{pool.device, set, vk_layout.handle, name};
}
void DescriptorPool::free(DescriptorSet set)
{
    if (current_sets > 0) vkFreeDescriptorSets(pool.device, pool.handle, 1, &set.set);
}

VkDescriptorSetLayout DescriptorPool::layout() { return vk_layout.handle; }

void DescriptorPool::update_descriptor_sets(DescriptorSet const& set,
                                            std::vector<DescriptorUseVector> const& uses)
{
    assert(uses.size() == bindings.size() && "Must use the correct number of descriptors");

    std::vector<DescriptorUse> descriptor_uses;
    for (size_t i = 0; i < bindings.size(); i++)
    {
        descriptor_uses.emplace_back(bindings[i].binding, bindings[i].descriptorCount,
                                     bindings[i].descriptorType, uses[i]);
    }
    std::vector<VkWriteDescriptorSet> write_descriptors;
    for (auto& use : descriptor_uses)
    {
        write_descriptors.push_back(use.get_write_descriptor_set(set.set));
    }

    vkUpdateDescriptorSets(pool.device, static_cast<uint32_t>(write_descriptors.size()),
                           write_descriptors.data(), 0, nullptr);
}

// Render Pass

VkRenderPass create_render_pass(VkDevice device, VkFormat swapchain_image_format,
                                VkFormat depth_image_format, std::string const& name)
{
    VkAttachmentDescription color_attachment{};
    color_attachment.format = swapchain_image_format;
    color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentDescription depth_attachment{};
    depth_attachment.format = depth_image_format;
    depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth_attachment.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    std::array<VkAttachmentDescription, 2> attach_desc{color_attachment, depth_attachment};

    std::array<VkAttachmentReference, 2> attach_refs{
        VkAttachmentReference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},           // color
        VkAttachmentReference{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL}};  // depth

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &attach_refs[0];
    subpass.pDepthStencilAttachment = &attach_refs[1];

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo render_pass_info{};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.attachmentCount = static_cast<uint32_t>(attach_desc.size());
    render_pass_info.pAttachments = attach_desc.data();
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass;
    render_pass_info.dependencyCount = 1;
    render_pass_info.pDependencies = &dependency;

    VkRenderPass render_pass;

    VK_CHECK_RESULT(vkCreateRenderPass(device, &render_pass_info, nullptr, &render_pass));
    debug_utils_helper.set_debug_object_name(VK_OBJECT_TYPE_RENDER_PASS, render_pass, name);
    return render_pass;
}

void destroy_render_pass(VkDevice device, VkRenderPass render_pass)
{
    vkDestroyRenderPass(device, render_pass, nullptr);
}

// Framebuffer

auto create_framebuffer(VkDevice device, VkRenderPass render_pass, VkExtent2D extent,
                        std::vector<VkImageView> image_views)
{
    VkFramebufferCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    create_info.renderPass = render_pass;
    create_info.attachmentCount = static_cast<uint32_t>(image_views.size());
    create_info.pAttachments = image_views.data();
    create_info.width = extent.width;
    create_info.height = extent.height;
    create_info.layers = 1;

    VkFramebuffer framebuffer;
    VK_CHECK_RESULT(vkCreateFramebuffer(device, &create_info, nullptr, &framebuffer))
    return HandleWrapper(device, framebuffer, vkDestroyFramebuffer);
}

Framebuffer::Framebuffer(VkDevice device, VkRenderPass render_pass, VkExtent2D extent,
                         std::vector<VkImageView> image_views, std::string const& name)
    : framebuffer(create_framebuffer(device, render_pass, extent, image_views))
{
    debug_utils_helper.set_debug_object_name(VK_OBJECT_TYPE_FRAMEBUFFER, framebuffer.handle, name);
}

// ShaderModule

auto create_shader_module(VkDevice device, std::vector<uint32_t> const& spirv_code)
{
    VkShaderModuleCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = static_cast<uint32_t>(spirv_code.size() * sizeof(uint32_t));
    create_info.pCode = spirv_code.data();

    VkShaderModule module;
    VK_CHECK_RESULT(vkCreateShaderModule(device, &create_info, nullptr, &module));
    return HandleWrapper(device, module, vkDestroyShaderModule);
}

ShaderModule::ShaderModule(VkDevice device, std::vector<uint32_t> const& spirv_code,
                           std::string const& name)
    : module(create_shader_module(device, spirv_code))
{
    debug_utils_helper.set_debug_object_name(VK_OBJECT_TYPE_SHADER_MODULE, module.handle, name);
}

PipelineBuilder::PipelineBuilder(VkDevice device, std::string const& source_folder)
    : device(device), source_folder(source_folder)
{
}

void PipelineBuilder::shutdown()
{
    for (auto& pipeline : graphics_pipelines) vkDestroyPipeline(device, pipeline.pipeline, nullptr);
    for (auto& pipeline : compute_pipelines) vkDestroyPipeline(device, pipeline.pipeline, nullptr);
    for (auto& layout : layouts) vkDestroyPipelineLayout(device, layout, nullptr);
}

VkPipeline PipelineBuilder::get_pipeline(GraphicsPipelineHandle const& handle)
{
    return graphics_pipelines.at(handle.index).pipeline;
}
VkPipeline PipelineBuilder::get_pipeline(ComputePipelineHandle const& handle)
{
    return compute_pipelines.at(handle.index).pipeline;
}

VkPipelineLayout PipelineBuilder::create_layout(
    std::vector<VkDescriptorSetLayout> const& descriptor_layouts,
    std::vector<VkPushConstantRange> const& push_constants, std::string const& name)
{
    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = static_cast<uint32_t>(descriptor_layouts.size());
    pipeline_layout_info.pSetLayouts = descriptor_layouts.data();
    pipeline_layout_info.pushConstantRangeCount = static_cast<uint32_t>(push_constants.size());
    pipeline_layout_info.pPushConstantRanges = push_constants.data();

    VkPipelineLayout layout;
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &layout));
    layouts.push_back(layout);
    debug_utils_helper.set_debug_object_name(VK_OBJECT_TYPE_PIPELINE_LAYOUT, layout, name);

    return layout;
}

GraphicsPipelineHandle PipelineBuilder::create_pipeline(GraphicsPipelineDetails const& details)
{
    uint32_t index = (uint32_t)graphics_pipelines.size();
    graphics_pipelines.push_back(details);
    graphics_pipelines.back().pipeline = create_immutable_pipeline(details);
    return {index};
}

ComputePipelineHandle PipelineBuilder::create_pipeline(ComputePipelineDetails const& details)
{
    uint32_t index = (uint32_t)compute_pipelines.size();
    compute_pipelines.push_back(details);
    compute_pipelines.back().pipeline = create_immutable_pipeline(details);
    return {index};
}

VkPipeline PipelineBuilder::create_immutable_pipeline(GraphicsPipelineDetails const& details)

{
    std::vector<uint32_t> vertex_code;
    if (details.spirv_vert_data.size() == 0)
    {
        vertex_code = load_spirv(details.vert_shader);
        assert(vertex_code.size());
    }
    std::vector<uint32_t> fragment_code;
    if (details.spirv_frag_data.size() == 0)
    {
        fragment_code = load_spirv(details.frag_shader);
        assert(fragment_code.size());
    }

    ShaderModule vertex_module(
        device, details.spirv_vert_data.size() == 0 ? vertex_code : details.spirv_vert_data,
        "vert_shader_for_" + details.name);
    ShaderModule fragment_module(
        device, details.spirv_frag_data.size() == 0 ? fragment_code : details.spirv_frag_data,
        "frag_shader_for_" + details.name);

    VkPipelineShaderStageCreateInfo vertex_shader_create_info{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        nullptr,
        0,
        VK_SHADER_STAGE_VERTEX_BIT,
        vertex_module.module.handle,
        "main"};
    VkPipelineShaderStageCreateInfo fragment_shader_create_info{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        nullptr,
        0,
        VK_SHADER_STAGE_FRAGMENT_BIT,
        fragment_module.module.handle,
        "main"};

    VkPipelineShaderStageCreateInfo shader_stages[] = {vertex_shader_create_info,
                                                       fragment_shader_create_info};

    VkPipelineVertexInputStateCreateInfo vertex_input_info{};
    vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input_info.vertexBindingDescriptionCount =
        static_cast<uint32_t>(details.binding_desc.size());
    vertex_input_info.pVertexBindingDescriptions = details.binding_desc.data();
    vertex_input_info.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(details.attribute_desc.size());
    vertex_input_info.pVertexAttributeDescriptions = details.attribute_desc.data();

    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    input_assembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(details.extent.width);  // use dynamic viewport
    viewport.height = static_cast<float>(details.extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = details.extent;

    VkPipelineViewportStateCreateInfo viewport_state{};
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.pViewports = &viewport;
    viewport_state.scissorCount = 1;
    viewport_state.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = details.polygon_mode;
    rasterizer.lineWidth = details.line_width;
    rasterizer.cullMode = details.cull_mode;
    rasterizer.frontFace = details.front_face;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth_stencil_info{};
    depth_stencil_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth_stencil_info.depthTestEnable = details.enable_depth;
    depth_stencil_info.depthWriteEnable = details.enable_depth;
    depth_stencil_info.depthCompareOp = VK_COMPARE_OP_LESS;
    depth_stencil_info.depthBoundsTestEnable = VK_FALSE;
    depth_stencil_info.stencilTestEnable = details.enable_stencil;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;
    if (details.enable_blending)
    {
        colorBlendAttachment.blendEnable = VK_TRUE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
    }

    VkPipelineColorBlendStateCreateInfo color_blending{};
    color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blending.logicOpEnable = VK_FALSE;
    color_blending.logicOp = VK_LOGIC_OP_COPY;
    color_blending.attachmentCount = 1;
    color_blending.pAttachments = &colorBlendAttachment;
    color_blending.blendConstants[0] = 0.0f;
    color_blending.blendConstants[1] = 0.0f;
    color_blending.blendConstants[2] = 0.0f;
    color_blending.blendConstants[3] = 0.0f;

    VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};

    VkPipelineDynamicStateCreateInfo dynamic_state{};
    dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic_state.dynamicStateCount = 2;
    dynamic_state.pDynamicStates = dynamic_states;

    VkGraphicsPipelineCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    create_info.stageCount = 2;
    create_info.pStages = shader_stages;
    create_info.pVertexInputState = &vertex_input_info;
    create_info.pInputAssemblyState = &input_assembly;
    create_info.pViewportState = &viewport_state;
    create_info.pRasterizationState = &rasterizer;
    create_info.pMultisampleState = &multisampling;
    create_info.pDepthStencilState = &depth_stencil_info;
    create_info.pColorBlendState = &color_blending;
    create_info.layout = details.pipeline_layout;
    create_info.renderPass = details.render_pass;
    create_info.subpass = 0;
    create_info.basePipelineHandle = VK_NULL_HANDLE;

    VkPipeline pipeline;
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, cache, 1, &create_info, nullptr, &pipeline));
    debug_utils_helper.set_debug_object_name(VK_OBJECT_TYPE_PIPELINE, pipeline, details.name);

    return pipeline;
}
VkPipeline PipelineBuilder::create_immutable_pipeline(ComputePipelineDetails const& details)
{
    auto compute_code = load_spirv(details.compute_shader);
    assert(compute_code.size());

    ShaderModule compute_module(device, compute_code, "compute_shader_for_" + details.name);

    VkPipelineShaderStageCreateInfo compute_shader_create_info{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        nullptr,
        0,
        VK_SHADER_STAGE_COMPUTE_BIT,
        compute_module.module.handle,
        "main"};

    VkComputePipelineCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    create_info.stage = compute_shader_create_info;
    create_info.layout = details.pipeline_layout;

    VkPipeline pipeline;
    VK_CHECK_RESULT(vkCreateComputePipelines(device, cache, 1, &create_info, nullptr, &pipeline));
    debug_utils_helper.set_debug_object_name(VK_OBJECT_TYPE_PIPELINE, pipeline, details.name);
    return pipeline;
}
void PipelineBuilder::recompile_pipelines()
{
    for (auto& details : graphics_pipelines)
    {
        vkDestroyPipeline(device, details.pipeline, nullptr);

        details.pipeline = create_immutable_pipeline(details);
    }
    for (auto& details : compute_pipelines)
    {
        vkDestroyPipeline(device, details.pipeline, nullptr);

        details.pipeline = create_immutable_pipeline(details);
    }
}

std::vector<uint32_t> PipelineBuilder::load_spirv(std::string const& filename) const
{
    std::string shader_path;
    if (source_folder != "")
        shader_path = source_folder + "/assets/shaders/" + filename;
    else
        shader_path = "assets/shaders/" + filename;
#ifdef WIN32
    std::string win_path;
    for (auto& c : shader_path)
    {
        if (c == '/')
            win_path.push_back('\\');
        else
            win_path.push_back(c);
    }
    shader_path = win_path;
#endif
    std::ifstream file(shader_path, std::ios::ate | std::ios::binary);
    if (!file.is_open())
    {
        fmt::print(stderr, "Failed to open file: {}\n", filename);
        return {};
    }
    size_t file_size = (size_t)file.tellg();
    std::vector<char> buffer(file_size);
    file.seekg(0);
    file.read(buffer.data(), file_size);

    std::vector<uint32_t> aligned_code(buffer.size() / 4);
    memcpy(aligned_code.data(), buffer.data(), buffer.size());

    return aligned_code;
}

// Memory
MemoryAllocator::MemoryAllocator(VkPhysicalDevice physical_device, VkDevice device)
    : physical_device(physical_device), device(device)
{
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
}

void MemoryAllocator::shutdown()
{
    image_allocations.clear();
    buffer_allocations.clear();
}

MemoryAllocator::Allocation<VkImage> MemoryAllocator::allocate_image(VkImage image,
                                                                     VkDeviceSize size,
                                                                     MemoryUsage usage)
{
    VkMemoryRequirements memory_requirements;
    vkGetImageMemoryRequirements(device, image, &memory_requirements);

    uint32_t memory_type =
        find_memory_type(memory_requirements.memoryTypeBits, get_memory_property_flags(usage));

    auto device_memory = create_device_memory(memory_requirements.size, memory_type);

    vkBindImageMemory(device, image, device_memory.handle, 0);

    InternalAllocation alloc{memory_requirements.size, std::move(device_memory)};
    image_allocations.emplace_back(image, std::move(alloc));
    return Allocation(this, image);
}

MemoryAllocator::Allocation<VkBuffer> MemoryAllocator::allocate_buffer(VkBuffer buffer,
                                                                       VkDeviceSize size,
                                                                       MemoryUsage usage)
{
    VkMemoryRequirements memory_requirements;
    vkGetBufferMemoryRequirements(device, buffer, &memory_requirements);

    uint32_t memory_type =
        find_memory_type(memory_requirements.memoryTypeBits, get_memory_property_flags(usage));

    auto device_memory = create_device_memory(memory_requirements.size, memory_type);

    vkBindBufferMemory(device, buffer, device_memory.handle, 0);

    InternalAllocation alloc{memory_requirements.size, std::move(device_memory)};
    buffer_allocations.emplace_back(buffer, std::move(alloc));

    return Allocation(this, buffer);
}

void MemoryAllocator::free(VkImage image)
{
    auto it = std::find_if(std::begin(image_allocations), std::end(image_allocations),
                           [&](auto const& elem) { return elem.first == image; });
    if (it != std::end(image_allocations))
    {
        image_allocations.erase(it);
    }
}
void MemoryAllocator::free(VkBuffer buffer)
{
    auto it = std::find_if(std::begin(buffer_allocations), std::end(buffer_allocations),
                           [&](auto const& elem) { return elem.first == buffer; });

    if (it != std::end(buffer_allocations))
    {
        buffer_allocations.erase(it);
    }
}
void MemoryAllocator::map(VkBuffer buffer, void** data_ptr)
{
    auto it = std::find_if(std::begin(buffer_allocations), std::end(buffer_allocations),
                           [&](auto const& elem) { return elem.first == buffer; });

    if (it != std::end(buffer_allocations))
    {
        vkMapMemory(device, it->second.memory.handle, 0, it->second.size, 0, data_ptr);
    }
}
void MemoryAllocator::unmap(VkBuffer buffer)
{
    auto it = std::find_if(std::begin(buffer_allocations), std::end(buffer_allocations),
                           [&](auto const& elem) { return elem.first == buffer; });

    if (it != std::end(buffer_allocations))
    {
        vkUnmapMemory(device, it->second.memory.handle);
    }
}

void MemoryAllocator::flush(VkBuffer buffer)
{
    auto it = std::find_if(std::begin(buffer_allocations), std::end(buffer_allocations),
                           [&](auto const& elem) { return elem.first == buffer; });

    if (it != std::end(buffer_allocations))
    {
        VkMappedMemoryRange range[1] = {};
        range[0].sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range[0].memory = it->second.memory.handle;
        range[0].size = it->second.size;
        VK_CHECK_RESULT(vkFlushMappedMemoryRanges(device, 1, range));
    }
}
VkMemoryPropertyFlags MemoryAllocator::get_memory_property_flags(MemoryUsage usage)
{
    switch (usage)
    {
        case MemoryUsage::gpu:
            return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        case MemoryUsage::cpu:
            return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        case MemoryUsage::cpu_to_gpu:
            return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        case MemoryUsage::gpu_to_cpu:
            return VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        case MemoryUsage::cpu_copy:
            return VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

        default:
            return 0;
    }
}

HandleWrapper<VkDeviceMemory, PFN_vkFreeMemory> MemoryAllocator::create_device_memory(
    VkDeviceSize max_size, uint32_t memory_type_index)
{
    VkMemoryAllocateInfo allocation_info{};
    allocation_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation_info.allocationSize = max_size;
    allocation_info.memoryTypeIndex = memory_type_index;

    VkDeviceMemory memory;
    VK_CHECK_RESULT(vkAllocateMemory(device, &allocation_info, nullptr, &memory));
    return HandleWrapper(device, memory, vkFreeMemory);
}

uint32_t MemoryAllocator::find_memory_type(uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++)
    {
        if ((typeFilter & (1 << i)) &&
            (memory_properties.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return i;
        }
    }
    assert(false && "failed to find suitable memory type!");
    return 0;
}
MemoryAllocator::Pool::Pool(VkDevice device, VkDeviceMemory device_memory, VkDeviceSize max_size)
    : device_memory(HandleWrapper(device, device_memory, vkFreeMemory)), max_size(max_size)
{
}

// Image

auto create_image(VkDevice device, VkFormat format, VkImageTiling image_tiling, VkExtent3D extent,
                  VkImageUsageFlags usage)
{
    VkImageCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    create_info.imageType = VK_IMAGE_TYPE_2D;
    create_info.extent = extent;
    create_info.mipLevels = 1;
    create_info.arrayLayers = 1;
    create_info.format = format;
    create_info.tiling = image_tiling;
    create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    create_info.usage = usage;
    create_info.samples = VK_SAMPLE_COUNT_1_BIT;
    create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkImage image;
    VK_CHECK_RESULT(vkCreateImage(device, &create_info, nullptr, &image));
    return HandleWrapper(device, image, vkDestroyImage);
}

auto create_image_view(VkDevice device, VkImage image, VkFormat format, VkImageAspectFlags aspects)
{
    VkImageViewCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    create_info.image = image;
    create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    create_info.format = format;
    create_info.subresourceRange.aspectMask = aspects;
    create_info.subresourceRange.baseMipLevel = 0;
    create_info.subresourceRange.levelCount = 1;
    create_info.subresourceRange.baseArrayLayer = 0;
    create_info.subresourceRange.layerCount = 1;

    VkImageView image_view;
    VK_CHECK_RESULT(vkCreateImageView(device, &create_info, nullptr, &image_view));
    return HandleWrapper(device, image_view, vkDestroyImageView);
}

auto create_sampler(VkDevice device)
{
    VkSamplerCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    create_info.magFilter = VK_FILTER_LINEAR;
    create_info.minFilter = VK_FILTER_LINEAR;
    create_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    create_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    create_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    create_info.anisotropyEnable = VK_TRUE;
    create_info.maxAnisotropy = 16.0f;
    create_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    create_info.unnormalizedCoordinates = VK_FALSE;
    create_info.compareEnable = VK_FALSE;
    create_info.compareOp = VK_COMPARE_OP_ALWAYS;
    create_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    VkSampler sampler;
    VK_CHECK_RESULT(vkCreateSampler(device, &create_info, nullptr, &sampler))
    return HandleWrapper(device, sampler, vkDestroySampler);
}

Image::Image(VkDevice device, MemoryAllocator& memory, Queue& queue, std::string const& name,
             VkFormat format, VkImageTiling tiling, uint32_t width, uint32_t height,
             VkImageUsageFlags usage, VkImageLayout layout, VkImageAspectFlags aspects,
             VkDeviceSize size, MemoryUsage memory_usage)
    : memory_ptr(&memory),
      image(create_image(device, format, tiling, {width, height, 1}, usage)),
      image_allocation(memory.allocate_image(image.handle, size, memory_usage)),
      image_view(create_image_view(device, image.handle, format, aspects)),
      sampler(create_sampler(device)),
      format(format),
      layout(layout),
      width(width),
      height(height)
{
    // no need to change layout
    if (layout == VK_IMAGE_LAYOUT_UNDEFINED) return;
    Fence fence(device, "image" + name + "_upload_fence");
    CommandBuffer cmd_buf(device, queue, "image" + name + "_upload_cmd_buf");
    cmd_buf.begin();
    set_image_layout(cmd_buf.get(), image.handle, VK_IMAGE_LAYOUT_UNDEFINED, layout,
                     {aspects, 0, 1, 0, 1});
    cmd_buf.end();
    queue.submit(cmd_buf, fence);
    fence.wait();

    debug_utils_helper.set_debug_object_name(VK_OBJECT_TYPE_IMAGE, image.handle, name);
    debug_utils_helper.set_debug_object_name(VK_OBJECT_TYPE_SAMPLER, sampler.handle,
                                             name + "_sampler");
    debug_utils_helper.set_debug_object_name(VK_OBJECT_TYPE_IMAGE_VIEW, image_view.handle,
                                             name + "_view");
}

VkDescriptorImageInfo Image::descriptor_info() const
{
    return {sampler.handle, image_view.handle, layout};
}

// Buffer

auto create_buffer(VkDevice device, VkDeviceSize size, VkBufferUsageFlags usage)
{
    VkBufferCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    create_info.size = size;
    create_info.usage = usage;
    create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer;
    VK_CHECK_RESULT(vkCreateBuffer(device, &create_info, nullptr, &buffer));

    return HandleWrapper(device, buffer, vkDestroyBuffer);
}

Buffer::Buffer(VkDevice device, MemoryAllocator& memory, std::string const& name,
               VkBufferUsageFlags usage, VkDeviceSize size, MemoryUsage memory_usage)
    : memory_ptr(&memory),
      buffer(create_buffer(device, size, usage)),
      buffer_allocation(memory.allocate_buffer(buffer.handle, size, memory_usage)),
      buf_size(size)
{
    debug_utils_helper.set_debug_object_name(VK_OBJECT_TYPE_BUFFER, buffer.handle, name);
}
void Buffer::map()
{
    memory_ptr->map(buffer.handle, &mapped_ptr);
    is_mapped = true;
}
void Buffer::unmap()
{
    memory_ptr->unmap(buffer.handle);
    is_mapped = false;
}
void Buffer::copy_to(void const* pData, size_t size)
{
    if (!is_mapped) map();

    if (mapped_ptr != nullptr) memcpy(mapped_ptr, pData, size);
}
void Buffer::copy_bytes(unsigned char* data, size_t size)
{
    if (!is_mapped) map();
    if (mapped_ptr != nullptr) memcpy(mapped_ptr, data, size);
}
void Buffer::flush() { memory_ptr->flush(buffer.handle); }

VkDeviceSize Buffer::size() const { return buf_size; }

VkDescriptorBufferInfo Buffer::descriptor_info() const { return {buffer.handle, 0, buf_size}; }

void bind_vertex_buffer(VkCommandBuffer command_buffer, Buffer const& buffer)
{
    VkDeviceSize offset = {0};
    VkBuffer buf = buffer.get();
    vkCmdBindVertexBuffers(command_buffer, 0, 1, &buf, &offset);
}

// Image Layout Transition

void set_image_layout(VkCommandBuffer cmdbuffer, VkImage image, VkImageLayout old_image_layout,
                      VkImageLayout new_image_layout, VkImageSubresourceRange subresource_range,
                      VkPipelineStageFlags src_stage_mask, VkPipelineStageFlags dst_stage_mask)
{
    // Create an image barrier object
    VkImageMemoryBarrier image_memory_barrier{};
    image_memory_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    image_memory_barrier.oldLayout = old_image_layout;
    image_memory_barrier.newLayout = new_image_layout;
    image_memory_barrier.image = image;
    image_memory_barrier.subresourceRange = subresource_range;

    // Source layouts (old)
    // Source access mask controls actions that have to be finished on the
    // old layout before it will be transitioned to the new layout
    switch (old_image_layout)
    {
        case VK_IMAGE_LAYOUT_UNDEFINED:
            // Image layout is undefined (or does not matter)
            // Only valid as initial layout
            // No flags required, listed only for completeness
            image_memory_barrier.srcAccessMask = 0;
            break;

        case VK_IMAGE_LAYOUT_PREINITIALIZED:
            // Image is preinitialized
            // Only valid as initial layout for linear images, preserves
            // memory contents Make sure host writes have been finished
            image_memory_barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            break;

        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            // Image is a color attachment
            // Make sure any writes to the color buffer have been finished
            image_memory_barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            break;

        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            // Image is a depth/stencil attachment
            // Make sure any writes to the depth/stencil buffer have been
            // finished
            image_memory_barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            break;

        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            // Image is a transfer source
            // Make sure any reads from the image have been finished
            image_memory_barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            break;

        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            // Image is a transfer destination
            // Make sure any writes to the image have been finished
            image_memory_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            break;

        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            // Image is read by a shader
            // Make sure any shader reads from the image have been finished
            image_memory_barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            break;
        default:
            // Other source layouts aren't handled (yet)
            break;
    }

    // Target layouts (new)
    // Destination access mask controls the dependency for the new image
    // layout
    switch (new_image_layout)
    {
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            // Image will be used as a transfer destination
            // Make sure any writes to the image have been finished
            image_memory_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            break;

        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            // Image will be used as a transfer source
            // Make sure any reads from the image have been finished
            image_memory_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            break;

        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            // Image will be used as a color attachment
            // Make sure any writes to the color buffer have been finished
            image_memory_barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            break;

        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            // Image layout will be used as a depth/stencil attachment
            // Make sure any writes to depth/stencil buffer have been
            // finished
            image_memory_barrier.dstAccessMask =
                image_memory_barrier.dstAccessMask | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            break;

        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            // Image will be read in a shader (sampler, input attachment)
            // Make sure any writes to the image have been finished
            if (image_memory_barrier.srcAccessMask == 0)
            {
                image_memory_barrier.srcAccessMask =
                    VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            }
            image_memory_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            break;
        default:
            // Other source layouts aren't handled (yet)
            break;
    }
    // Put barrier inside setup command buffer
    vkCmdPipelineBarrier(cmdbuffer, src_stage_mask, dst_stage_mask, 0, 0, nullptr, 0, nullptr, 1,
                         &image_memory_barrier);
}

VkFormat get_depth_image_format(VkPhysicalDevice phys_device)
{
    std::array<VkFormat, 2> depth_formats{VK_FORMAT_D32_SFLOAT_S8_UINT,
                                          VK_FORMAT_D24_UNORM_S8_UINT};
    for (VkFormat format : depth_formats)
    {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(phys_device, format, &props);

        if ((props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT))
        {
            return format;
        }
    }
    return VK_FORMAT_UNDEFINED;
}

}  // namespace VK
