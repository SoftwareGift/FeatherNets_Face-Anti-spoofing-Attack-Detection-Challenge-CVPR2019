/*
 * vk_device.cpp - vulkan device
 *
 *  Copyright (c) 2018 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#include "vulkan_common.h"
#include "vk_device.h"
#include "vk_shader.h"
#include "vk_instance.h"
#include "vk_sync.h"
#include "vk_cmdbuf.h"
#include "file_handle.h"

namespace XCam {

SmartPtr<VKDevice> VKDevice::_default_dev;
Mutex VKDevice::_default_mutex;

VKDevice::~VKDevice ()
{
    if (_dev_id)
        vkDestroyDevice (_dev_id, _allocator.ptr ());
}

VKDevice::VKDevice (VkDevice id, const SmartPtr<VKInstance> &instance)
    : _dev_id (id)
    , _instance (instance)
{
    XCAM_ASSERT (instance.ptr ());
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (id));
    _allocator = instance->get_allocator ();
}

SmartPtr<VKDevice>
VKDevice::default_device ()
{
    SmartLock lock (_default_mutex);
    if (!_default_dev.ptr()) {
        _default_dev = create_device ();
    }
    XCAM_FAIL_RETURN (
        ERROR, _default_dev.ptr (), NULL,
        "VKDevice prepare default device failed.");
    return _default_dev;
}

SmartPtr<VKDevice>
VKDevice::create_device ()
{
    SmartPtr<VKInstance> instance = VKInstance::get_instance ();
    XCAM_FAIL_RETURN (
        ERROR, instance.ptr (), NULL,
        "vk create device failed");

    VkPhysicalDevice phy_dev = instance->get_physical_dev ();
    uint32_t compute_idx = instance->get_compute_queue_family_idx ();
    SmartPtr<VkAllocationCallbacks> allocator = instance->get_allocator ();

    float priority = 1.0f; //TODO, queue priority change?
    VkDeviceQueueCreateInfo dev_queue_info = {};
    dev_queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    dev_queue_info.queueFamilyIndex = compute_idx; // default use compute idx
    dev_queue_info.queueCount = 1;
    dev_queue_info.pQueuePriorities = &priority;

    VkDeviceCreateInfo dev_create_info = {};
    dev_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    //TODO, add graphics queue info
    dev_create_info.queueCreateInfoCount = 1;
    dev_create_info.pQueueCreateInfos = &dev_queue_info;

    VkDevice dev_id = 0;
    XCAM_VK_CHECK_RETURN (
        ERROR,
        vkCreateDevice (phy_dev, &dev_create_info, allocator.ptr (), &dev_id),
        NULL, "create vk device failed");

    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (dev_id));
    SmartPtr<VKDevice> device = new VKDevice (dev_id, instance);
    XCAM_ASSERT (device.ptr ());

    XCamReturn ret = device->prepare_compute_queue ();
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), NULL,
        "VKDevice prepare compute queue failed.");

    return device;
}

XCamReturn
VKDevice::prepare_compute_queue ()
{
    uint32_t compute_idx = _instance->get_compute_queue_family_idx ();
    vkGetDeviceQueue (_dev_id, compute_idx, 0, &_compute_queue);
    return XCAM_RETURN_NO_ERROR;
}

SmartPtr<VKShader>
VKDevice::create_shader (const char *file_name)
{
    FileHandle file (file_name, "rb");
    XCAM_FAIL_RETURN (
        ERROR, file.is_valid (), NULL,
        "VKDevice load shader failed when opend shader file:%s.",
        XCAM_STR (file_name));

    size_t file_size;
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (file.get_file_size (file_size)) || file_size == 0, NULL,
        "VKDevice load shader failed when read shader file:%s.",
        XCAM_STR (file_name));
    std::vector<uint32_t> content (XCAM_ALIGN_UP (file_size, 4) / 4, 0);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (file.read_file ((void *)content.data (), file_size)), NULL,
        "VKDevice load shader failed when read shader file:%s.",
        XCAM_STR (file_name));
    file.close ();

    SmartPtr<VKShader> shader = create_shader (content);
    if (shader.ptr ())
        shader->set_name (file_name);
    return shader;
}

SmartPtr<VKShader>
VKDevice::create_shader (const std::vector<uint32_t> &binary)
{
    XCAM_FAIL_RETURN (
        ERROR, XCAM_IS_VALID_VK_ID (_dev_id), NULL,
        "VKDevice load shader failed with error of null device ready.");
    XCAM_FAIL_RETURN (
        ERROR, binary.size () > 5, NULL,
        "VKDevice load shader failed since binary is corrupt.");

    VkShaderModule shader_id;
    VkShaderModuleCreateInfo module_create_info = {};
    module_create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_create_info.pNext = NULL;
    module_create_info.codeSize = binary.size() * sizeof (binary[0]);
    module_create_info.pCode = binary.data();
    module_create_info.flags = 0;

    XCAM_VK_CHECK_RETURN (
        ERROR, vkCreateShaderModule (_dev_id, &module_create_info, NULL, &shader_id),
        NULL, "VKDevice create shader module failed.");

    XCAM_IS_VALID_VK_ID (shader_id);
    return new VKShader (this, shader_id);
}

void
VKDevice::destroy_shader_id (VkShaderModule shader)
{
    if (XCAM_IS_VALID_VK_ID(_dev_id) && XCAM_IS_VALID_VK_ID (shader))
        vkDestroyShaderModule (_dev_id, shader, _allocator.ptr());
}

VkDeviceMemory
VKDevice::allocate_mem_id (VkDeviceSize size, VkMemoryPropertyFlags memory_prop)
{
    VkDeviceMemory mem_id;
    VkMemoryAllocateInfo mem_alloc_info = {};
    mem_alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mem_alloc_info.allocationSize = size;
    mem_alloc_info.memoryTypeIndex = _instance->get_mem_type_index (memory_prop);

    XCAM_FAIL_RETURN (
        ERROR, mem_alloc_info.memoryTypeIndex != (uint32_t)(-1), VK_NULL_HANDLE,
        "VKDevice create mem id failed, can NOT find memory type:0x%08x.", (uint32_t)memory_prop);

    XCAM_VK_CHECK_RETURN (
        ERROR, vkAllocateMemory (_dev_id, &mem_alloc_info, _allocator.ptr (), &mem_id),
        VK_NULL_HANDLE, "create vk buffer failed in allocating memory");
    return mem_id;
}

void
VKDevice::free_mem_id (VkDeviceMemory mem)
{
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (_dev_id));
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (mem));

    vkFreeMemory (_dev_id, mem, _allocator.ptr ());
}

XCamReturn
VKDevice::map_mem (VkDeviceMemory mem, VkDeviceSize size, VkDeviceSize offset, void *&ptr)
{
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (mem));
    XCAM_VK_CHECK_RETURN (
        ERROR, vkMapMemory (_dev_id, mem, offset, size, 0, &ptr), XCAM_RETURN_ERROR_VULKAN,
        "vk device map mem failed. size:%lld", size);
    return XCAM_RETURN_NO_ERROR;
}

void
VKDevice::unmap_mem (VkDeviceMemory mem)
{
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (mem));
    vkUnmapMemory (_dev_id, mem);
}

VkBuffer
VKDevice::create_buf_id (VkBufferUsageFlags usage, uint32_t size)
{
    VkBufferCreateInfo buf_create_info = {};
    buf_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_create_info.size = size;
    buf_create_info.usage = usage;
    buf_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buf_id;
    XCAM_VK_CHECK_RETURN (
        ERROR, vkCreateBuffer (_dev_id, &buf_create_info, _allocator.ptr (), &buf_id),
        VK_NULL_HANDLE, "create vk buffer failed");

    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (buf_id));
    return buf_id;
}

void
VKDevice::destroy_buf_id (VkBuffer buf)
{
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (buf));

    vkDestroyBuffer (_dev_id, buf, _allocator.ptr ());
}

XCamReturn
VKDevice::bind_buffer (VkBuffer buf, VkDeviceMemory mem, VkDeviceSize offset)
{
    XCAM_VK_CHECK_RETURN (
        ERROR, vkBindBufferMemory (_dev_id, buf, mem, offset),
        XCAM_RETURN_ERROR_VULKAN, "vkdevice bind buffer to mem failed");

    return XCAM_RETURN_NO_ERROR;
}

VkDescriptorPool
VKDevice::create_desc_pool (const VkDescriptorPoolCreateInfo &info)
{
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (_dev_id));

    VkDescriptorPool pool_id;
    XCAM_VK_CHECK_RETURN (
        ERROR,
        vkCreateDescriptorPool (_dev_id, &info, _allocator.ptr (), &pool_id),
        VK_NULL_HANDLE,
        "vkdevice create desriptor pool failed");
    return pool_id;
}

void
VKDevice::destroy_desc_pool (VkDescriptorPool pool)
{
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (_dev_id));
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (pool));
    vkDestroyDescriptorPool (_dev_id, pool, _allocator.ptr ());
}

VkDescriptorSet
VKDevice::allocate_desc_set (const VkDescriptorSetAllocateInfo &info)
{
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (_dev_id));

    VkDescriptorSet set_id;
    XCAM_VK_CHECK_RETURN (
        ERROR,
        vkAllocateDescriptorSets (_dev_id, &info, &set_id),
        VK_NULL_HANDLE,
        "vkdevice create desriptor set failed");
    return set_id;

}

XCamReturn
VKDevice::free_desc_set (VkDescriptorSet set, VkDescriptorPool pool)
{
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (_dev_id));
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (set));
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (pool));

    XCAM_VK_CHECK_RETURN (
        ERROR,
        vkFreeDescriptorSets (_dev_id, pool, 1, &set),
        XCAM_RETURN_ERROR_VULKAN,
        "vkdevice free desriptor set from pool failed");
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
VKDevice::update_desc_set (const std::vector<VkWriteDescriptorSet> &sets)
{
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (_dev_id));
    vkUpdateDescriptorSets (_dev_id, sets.size (), sets.data (), 0, NULL);

    return XCAM_RETURN_NO_ERROR;
}

VkCommandPool
VKDevice::create_cmd_pool (VkFlags queue_flag)
{
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (_dev_id));
    XCAM_ASSERT (_instance.ptr ());
    VkCommandPool pool_id = VK_NULL_HANDLE;

    VkCommandPoolCreateInfo create_pool_info = {};
    create_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    create_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (queue_flag == VK_QUEUE_COMPUTE_BIT)
        create_pool_info.queueFamilyIndex = _instance->get_compute_queue_family_idx ();
    else if (queue_flag == VK_QUEUE_GRAPHICS_BIT)
        create_pool_info.queueFamilyIndex = _instance->get_graphics_queue_family_idx ();
    else {
        XCAM_LOG_WARNING ("VKDevice create command pool failed, queue_flag(%d) not supported.", queue_flag);
        return VK_NULL_HANDLE;
    }

    XCAM_VK_CHECK_RETURN (
        ERROR, vkCreateCommandPool (_dev_id, &create_pool_info,  _allocator.ptr (), &pool_id),
        VK_NULL_HANDLE, "VKDevice create command pool failed.");
    return pool_id;
}

void
VKDevice::destroy_cmd_pool (VkCommandPool pool)
{
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (_dev_id));
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (pool));
    vkDestroyCommandPool (_dev_id, pool, _allocator.ptr ());
}

VkCommandBuffer
VKDevice::allocate_cmd_buffer (VkCommandPool pool)
{
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (_dev_id));
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (pool));

    VkCommandBufferAllocateInfo allocate_info = {};
    allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocate_info.commandPool = pool;
    allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate_info.commandBufferCount = 1;

    VkCommandBuffer buf_id = VK_NULL_HANDLE;

    XCAM_VK_CHECK_RETURN (
        ERROR, vkAllocateCommandBuffers (_dev_id, &allocate_info, &buf_id),
        VK_NULL_HANDLE, "VKDevice create command buffers failed.");
    return buf_id;
}

void
VKDevice::free_cmd_buffer (VkCommandPool pool, VkCommandBuffer buf)
{
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (_dev_id));
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (pool));
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (buf));

    vkFreeCommandBuffers (_dev_id, pool, 1, &buf);
}

SmartPtr<VKFence>
VKDevice::create_fence (VkFenceCreateFlags flags)
{
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (_dev_id));

    VkFenceCreateInfo fence_info = {};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = flags;

    VkFence fence_id = VK_NULL_HANDLE;
    XCAM_VK_CHECK_RETURN (
        ERROR, vkCreateFence (_dev_id, &fence_info,  _allocator.ptr (), &fence_id),
        NULL, "VKDevice create fence failed.");
    return new VKFence (this, fence_id);
}

void
VKDevice::destroy_fence (VkFence fence)
{
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (_dev_id));
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (fence));

    vkDestroyFence (_dev_id, fence, _allocator.ptr ());
}

XCamReturn
VKDevice::reset_fence (VkFence fence)
{
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (_dev_id));
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (fence));

    XCAM_VK_CHECK_RETURN (
        ERROR, vkResetFences (_dev_id, 1,  &fence),
        XCAM_RETURN_ERROR_VULKAN, "VKDevice reset fence failed.");
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
VKDevice::wait_for_fence (VkFence fence, uint64_t timeout)
{
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (_dev_id));
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (fence));

    VkResult ret = vkWaitForFences (_dev_id, 1,  &fence, VK_TRUE, timeout);
    if (ret == VK_TIMEOUT) {
        XCAM_LOG_DEBUG ("VKDevice wait for fence timeout");
        return XCAM_RETURN_ERROR_TIMEOUT;
    }

    XCAM_FAIL_RETURN (
        ERROR, ret == VK_SUCCESS,
        XCAM_RETURN_ERROR_VULKAN, "VKDevice wait for fence failed.");
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
VKDevice::compute_queue_submit (const SmartPtr<VKCmdBuf> cmd_buf, const SmartPtr<VKFence> fence)
{
    XCAM_FAIL_RETURN (
        ERROR, cmd_buf.ptr (),
        XCAM_RETURN_ERROR_PARAM, "VKDevice compute queue submit failed, cmd_buf is empty.");

    VkCommandBuffer buf_id = cmd_buf->get_cmd_buf_id ();
    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &buf_id;

    VkFence fence_id = VK_NULL_HANDLE;
    if (fence.ptr ())
        fence_id = fence->get_fence_id ();
    XCAM_VK_CHECK_RETURN (
        ERROR, vkQueueSubmit (_compute_queue, 1, &submit_info, fence_id),
        XCAM_RETURN_ERROR_VULKAN, "VKDevice compute queue submit failed.");

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
VKDevice::compute_queue_wait_idle ()
{
    XCAM_FAIL_RETURN (
        ERROR, XCAM_IS_VALID_VK_ID (_compute_queue),
        XCAM_RETURN_ERROR_PARAM, "VKDevice compute queue wait idle failed, queue_id is null");

    XCAM_VK_CHECK_RETURN (
        ERROR, vkQueueWaitIdle (_compute_queue),
        XCAM_RETURN_ERROR_VULKAN, "VKDevice compute queue wait idle failed");

    return XCAM_RETURN_NO_ERROR;
}

}
