#pragma once

#include <cstdio>
#include <cassert>
#include <cstdint>

#include <string>
#include <mutex>
#include <variant>
#include <vector>
#include <utility>

#include <vulkan/vulkan.h>

#include <fmt/core.h>

const char* error_str(const VkResult result);
#define VK_CHECK_RESULT(f)                                                                  \
    {                                                                                       \
        VkResult res = (f);                                                                 \
        if (res != VK_SUCCESS)                                                              \
        {                                                                                   \
            fmt::print(stderr, "Fatal : VkResult is {} in {} at line {}\n", error_str(res), \
                       __FILE__, __LINE__);                                                 \
            assert(res > VK_SUCCESS); /*only crash on negative results */                   \
        }                                                                                   \
    }

namespace VK
{
struct DebugUtilHelper
{
    DebugUtilHelper() {}

    DebugUtilHelper(VkDevice device) : device(device)
    {
        SetDebugUtilsObjectNameEXT = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
            vkGetDeviceProcAddr(device, "vkSetDebugUtilsObjectNameEXT"));
        SetDebugUtilsObjectTagEXT = reinterpret_cast<PFN_vkSetDebugUtilsObjectTagEXT>(
            vkGetDeviceProcAddr(device, "vkSetDebugUtilsObjectTagEXT"));
    }

    template <typename T>
    void set_debug_object_name(VkObjectType type, T handle, std::string const& name)
    {
        if (name == "" || !SetDebugUtilsObjectNameEXT) return;

        VkDebugUtilsObjectNameInfoEXT info{};
        info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
        info.objectType = type;
        info.objectHandle = (uint64_t)handle;
        info.pObjectName = name.c_str();

        SetDebugUtilsObjectNameEXT(device, &info);
    }

private:
    VkDevice device;
    PFN_vkSetDebugUtilsObjectNameEXT SetDebugUtilsObjectNameEXT = nullptr;
    PFN_vkSetDebugUtilsObjectTagEXT SetDebugUtilsObjectTagEXT = nullptr;
};

extern DebugUtilHelper debug_utils_helper;
void setup_debug_util_helper(VkDevice device);

constexpr uint32_t FLAGS_NONE = 0;

template <typename T, typename Deleter>
class HandleWrapper
{
public:
    explicit HandleWrapper(VkDevice device, T handle, Deleter deleter)
        : device(device), handle(handle), deleter(deleter)
    {
    }
    ~HandleWrapper()
    {
        if (handle != nullptr)
        {
            deleter(device, handle, nullptr);
        }
    };
    HandleWrapper(HandleWrapper const& other) = delete;
    HandleWrapper& operator=(HandleWrapper const& other) = delete;

    HandleWrapper(HandleWrapper&& other) noexcept
        : device(other.device), handle(other.handle), deleter(other.deleter)
    {
        other.handle = nullptr;
    }
    HandleWrapper& operator=(HandleWrapper&& other) noexcept
    {
        if (this != &other)
        {
            if (handle != nullptr)
            {
                deleter(device, handle, nullptr);
            }

            device = other.device;
            handle = other.handle;
            deleter = other.deleter;
            other.handle = nullptr;
        }
        return *this;
    }

    VkDevice device;
    T handle;
    Deleter deleter;
};

class Fence
{
public:
    explicit Fence(VkDevice device, std::string const& name, VkFenceCreateFlags flags = 0);

    bool check() const;
    void wait(bool condition = true) const;
    void reset() const;
    VkFence get() const;

private:
    HandleWrapper<VkFence, PFN_vkDestroyFence> fence;
};

class Semaphore
{
public:
    explicit Semaphore(VkDevice device, std::string const& name);

    VkSemaphore get() const;

private:
    HandleWrapper<VkSemaphore, PFN_vkDestroySemaphore> semaphore;
};

class CommandBuffer;

class Queue
{
public:
    explicit Queue(VkDevice device, uint32_t family, std::string const& name,
                   uint32_t queue_index = 0);

    void submit(CommandBuffer const& command_buffer, Fence& fence);
    void submit(CommandBuffer const& command_buffer, Fence const& fence,
                Semaphore const& wait_semaphore, Semaphore const& signal_semaphore,
                VkPipelineStageFlags const stage_mask);

    void wait_idle();
    VkResult presentation_submit(VkPresentInfoKHR present_info);

    VkQueue get() const;
    int get_family() const;

private:
    void submit(VkSubmitInfo const& submitInfo, Fence const& fence);

    std::mutex submit_mutex;
    VkQueue queue;
    int queue_family;
};

class CommandPool
{
public:
    explicit CommandPool(VkDevice device, Queue const& queue, std::string const& name,
                         VkCommandPoolCreateFlags flags = 0);

    VkCommandBuffer allocate();
    void free(VkCommandBuffer command_buffer);

private:
    HandleWrapper<VkCommandPool, PFN_vkDestroyCommandPool> pool;
};

class CommandBuffer
{
public:
    explicit CommandBuffer(VkDevice device, Queue const& queue, std::string const& name);

    ~CommandBuffer();

    CommandBuffer(CommandBuffer const& other) = delete;
    CommandBuffer& operator=(CommandBuffer const& other) = delete;
    CommandBuffer(CommandBuffer&& other) noexcept;
    CommandBuffer& operator=(CommandBuffer&& other) noexcept;

    VkCommandBuffer get() const;

    void begin(VkCommandBufferUsageFlags flags = 0);
    void end();
    void reset();

private:
    VkDevice device;
    Queue const* queue;
    CommandPool pool;
    VkCommandBuffer command_buffer;
};

class SyncResources
{
public:
    explicit SyncResources(VkDevice device, Queue& graphics_queue, Queue& present_queue,
                           VkSwapchainKHR swapchain);

    void submit();
    VkResult present(uint32_t image_index);

    Queue& graphics_queue;
    Queue& present_queue;
    VkSwapchainKHR swapchain;

    Semaphore image_avail_sem;
    Semaphore render_finish_sem;

    Fence command_fence;
    CommandBuffer command_buffer;
};

using DescriptorUseVector =
    std::variant<std::vector<VkDescriptorBufferInfo>, std::vector<VkDescriptorImageInfo>,
                 std::vector<VkBufferView>>;

class DescriptorUse
{
public:
    explicit DescriptorUse(uint32_t bind_point, uint32_t count, VkDescriptorType type,
                           DescriptorUseVector descriptor_use_data);

    VkWriteDescriptorSet get_write_descriptor_set(VkDescriptorSet set);

    uint32_t bind_point;
    uint32_t count;
    VkDescriptorType type = VkDescriptorType::VK_DESCRIPTOR_TYPE_MAX_ENUM;
    DescriptorUseVector descriptor_use_data;
};

class DescriptorSet
{
public:
    explicit DescriptorSet(VkDevice device, VkDescriptorSet set, VkDescriptorSetLayout layout,
                           std::string const& name);

    void update(std::vector<DescriptorUse> descriptors) const;
    void bind(VkCommandBuffer cmdBuf, VkPipelineBindPoint bind_point, VkPipelineLayout layout,
              uint32_t location) const;

    VkDevice device;
    VkDescriptorSet set;
    VkDescriptorSetLayout layout;
};

class DescriptorPool
{
public:
    explicit DescriptorPool(VkDevice device,
                            std::vector<VkDescriptorSetLayoutBinding> const& bindings,
                            uint32_t count, std::string const& name);

    DescriptorSet allocate(std::string const& name);
    void free(DescriptorSet set);

    VkDescriptorSetLayout layout();

    void update_descriptor_sets(DescriptorSet const& set,
                                std::vector<DescriptorUseVector> const& uses);

private:
    HandleWrapper<VkDescriptorSetLayout, PFN_vkDestroyDescriptorSetLayout> vk_layout;
    HandleWrapper<VkDescriptorPool, PFN_vkDestroyDescriptorPool> pool;
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    uint32_t max_sets = 0;
    uint32_t current_sets = 0;
};

VkRenderPass create_render_pass(VkDevice device, VkFormat swapchain_image_format,
                                VkFormat depth_image_format, std::string const& name);
void destroy_render_pass(VkDevice device, VkRenderPass render_pass);
struct Framebuffer
{
    explicit Framebuffer(VkDevice device, VkRenderPass render_pass, VkExtent2D extent,
                         std::vector<VkImageView> image_views, std::string const& name);

    HandleWrapper<VkFramebuffer, PFN_vkDestroyFramebuffer> framebuffer;
};

struct ShaderModule
{
    explicit ShaderModule(VkDevice device, std::vector<uint32_t> const& spirv_code,
                          std::string const& name);

    HandleWrapper<VkShaderModule, PFN_vkDestroyShaderModule> module;
};

std::vector<uint32_t> load_spirv(std::string const& filename);

struct GraphicsPipelineDetails
{
    std::string name;
    VkPipeline pipeline{};
    VkPipelineLayout pipeline_layout{};

    std::string vert_shader;
    std::string frag_shader;
    std::vector<uint32_t> spirv_vert_data;
    std::vector<uint32_t> spirv_frag_data;

    VkRenderPass render_pass{};
    VkExtent2D extent{};

    std::vector<VkVertexInputBindingDescription> binding_desc;
    std::vector<VkVertexInputAttributeDescription> attribute_desc;
    bool enable_blending = false;
    VkPolygonMode polygon_mode = VK_POLYGON_MODE_FILL;
    float line_width = 1.0f;
    VkCullModeFlags cull_mode = VK_CULL_MODE_FRONT_BIT;
    VkFrontFace front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    bool enable_depth = false;
    bool enable_stencil = false;
};

struct ComputePipelineDetails
{
    std::string name;
    VkPipeline pipeline;
    VkPipelineLayout pipeline_layout;

    std::string compute_shader;
};

struct GraphicsPipelineHandle
{
    uint32_t index;
};

struct ComputePipelineHandle
{
    uint32_t index;
};

struct PipelineBuilder
{
    PipelineBuilder() {}  // Must give it the device before using it.
    explicit PipelineBuilder(VkDevice device, std::string const& source_folder);
    void shutdown();

    VkPipeline get_pipeline(GraphicsPipelineHandle const& handle);
    VkPipeline get_pipeline(ComputePipelineHandle const& handle);

    VkPipelineLayout create_layout(std::vector<VkDescriptorSetLayout> const& descriptor_layouts,
                                   std::vector<VkPushConstantRange> const& push_constants,
                                   std::string const& name);

    GraphicsPipelineHandle create_pipeline(GraphicsPipelineDetails const& details);
    ComputePipelineHandle create_pipeline(ComputePipelineDetails const& details);

    VkPipeline create_immutable_pipeline(GraphicsPipelineDetails const& details);
    VkPipeline create_immutable_pipeline(ComputePipelineDetails const& details);

    void recompile_pipelines();

private:
    VkDevice device = nullptr;
    VkPipelineCache cache = nullptr;
    std::string source_folder = "";

    std::vector<VkPipelineLayout> layouts;
    std::vector<GraphicsPipelineDetails> graphics_pipelines;
    std::vector<ComputePipelineDetails> compute_pipelines;

    std::vector<uint32_t> load_spirv(std::string const& filename) const;
};

enum class MemoryUsage
{
    gpu,
    cpu,
    cpu_to_gpu,
    gpu_to_cpu,
    cpu_copy
};

class MemoryAllocator
{
public:
    MemoryAllocator() {}
    explicit MemoryAllocator(VkPhysicalDevice physical_device, VkDevice device);

    void shutdown();

    template <typename T>
    struct Allocation
    {
        Allocation(MemoryAllocator* memory_ptr, T data) : memory_ptr(memory_ptr), data(data) {}
        ~Allocation()
        {
            if (data != VK_NULL_HANDLE) memory_ptr->free(data);
        }

        Allocation(Allocation const& other) noexcept = delete;
        Allocation& operator=(Allocation const& other) noexcept = delete;

        Allocation(Allocation&& other) noexcept : memory_ptr(other.memory_ptr), data(other.data)
        {
            other.data = VK_NULL_HANDLE;
        }
        Allocation& operator=(Allocation&& other) noexcept
        {
            if (this != &other)
            {
                if (data != VK_NULL_HANDLE) memory_ptr->free(data);
                memory_ptr = other.memory_ptr;
                data = other.data;
                other.data = VK_NULL_HANDLE;
            }
            return *this;
        }

        MemoryAllocator* memory_ptr;
        T data;
    };

    Allocation<VkImage> allocate_image(VkImage image, VkDeviceSize size, MemoryUsage usage);
    Allocation<VkBuffer> allocate_buffer(VkBuffer buffer, VkDeviceSize size, MemoryUsage usage);

    void free(VkImage image);
    void free(VkBuffer buffer);

    void map(VkBuffer buffer, void** data_ptr);
    void unmap(VkBuffer buffer);

    void flush(VkBuffer buffer);

private:
    // Unused currently
    struct Pool
    {
        Pool(VkDevice device, VkDeviceMemory device_memory, VkDeviceSize max_size);
        HandleWrapper<VkDeviceMemory, PFN_vkFreeMemory> device_memory;
        VkDeviceSize max_size;
        struct Allocation
        {
            bool allocated = false;
            VkDeviceSize size;
            VkDeviceSize offset;
        };
        std::vector<Allocation> allocations;
    };

    VkPhysicalDevice physical_device;
    VkDevice device;
    VkPhysicalDeviceMemoryProperties memory_properties;

    struct InternalAllocation
    {
        VkDeviceSize size;
        HandleWrapper<VkDeviceMemory, PFN_vkFreeMemory> memory;
    };

    std::vector<std::pair<VkImage, InternalAllocation>> image_allocations;
    std::vector<std::pair<VkBuffer, InternalAllocation>> buffer_allocations;

    VkMemoryPropertyFlags get_memory_property_flags(MemoryUsage usage);

    uint32_t find_memory_type(uint32_t typeFilter, VkMemoryPropertyFlags properties);

    HandleWrapper<VkDeviceMemory, PFN_vkFreeMemory> create_device_memory(VkDeviceSize max_size,
                                                                         uint32_t memory_type);
};

class Image
{
public:
    explicit Image(VkDevice device, MemoryAllocator& memory, Queue& queue, std::string const& name,
                   VkFormat format, VkImageTiling tiling, uint32_t width, uint32_t height,
                   VkImageUsageFlags usage, VkImageLayout layout, VkImageAspectFlags aspects,
                   VkDeviceSize size, MemoryUsage memory_usage);

    VkImage get() const { return image.handle; }
    VkDescriptorImageInfo descriptor_info() const;

    MemoryAllocator* memory_ptr;
    HandleWrapper<VkImage, PFN_vkDestroyImage> image;
    MemoryAllocator::Allocation<VkImage> image_allocation;
    HandleWrapper<VkImageView, PFN_vkDestroyImageView> image_view;
    HandleWrapper<VkSampler, PFN_vkDestroySampler> sampler;

    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    uint32_t width;
    uint32_t height;
};

class Buffer
{
public:
    explicit Buffer(VkDevice device, MemoryAllocator& memory, std::string const& name,
                    VkBufferUsageFlags usage, VkDeviceSize size, MemoryUsage memory_usage);

    VkBuffer get() const { return buffer.handle; }

    void map();
    void unmap();

    template <typename T>
    void copy_to(std::vector<T> const& data)
    {
        copy_to(reinterpret_cast<void const*>(data.data()), sizeof(T) * data.size());
    }

    template <typename T>
    void copy_to(T const& data)
    {
        copy_to(reinterpret_cast<void const*>(&data), sizeof(T));
    }

    void copy_bytes(unsigned char* data, size_t size);

    void flush();

    VkDescriptorBufferInfo descriptor_info() const;
    VkDeviceSize size() const;

private:
    MemoryAllocator* memory_ptr;
    HandleWrapper<VkBuffer, PFN_vkDestroyBuffer> buffer;
    MemoryAllocator::Allocation<VkBuffer> buffer_allocation;

    VkDeviceSize buf_size;
    bool is_mapped = false;
    void* mapped_ptr = nullptr;

    void copy_to(void const* pData, size_t size);
};

void bind_vertex_buffer(VkCommandBuffer command_buffer, Buffer const& buffer);

void set_image_layout(VkCommandBuffer command_buffer, VkImage image, VkImageLayout old_image_layout,
                      VkImageLayout new_image_layout, VkImageSubresourceRange subresource_range,
                      VkPipelineStageFlags src_stage_mask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                      VkPipelineStageFlags dst_stage_mask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

VkFormat get_depth_image_format(VkPhysicalDevice device);
}  // namespace VK
