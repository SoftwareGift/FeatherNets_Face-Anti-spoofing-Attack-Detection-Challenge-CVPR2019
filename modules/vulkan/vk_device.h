/*
 * vk_device.h - vulkan device
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

#ifndef XCAM_VK_DEVICE_H
#define XCAM_VK_DEVICE_H

#include <vulkan/vulkan_std.h>
#include <xcam_mutex.h>

namespace XCam {

class VKPipeline;
class VKShader;
class VKFence;
class VKCmdBuf;
class VKInstance;
class VKMemory;
class VKBuffer;

namespace VKDescriptor {
class Pool;
class Set;
};

class VKDevice
    : public RefObj
{
    friend class VKFence;
    friend class VKShader;
    friend class VKPipeline;
    friend class VKCmdBuf;
    friend class VKDescriptor::Pool;
    friend class VKDescriptor::Set;
    friend class VKMemory;
    friend class VKBuffer;
public:
    ~VKDevice ();
    static SmartPtr<VKDevice> default_device ();
    static SmartPtr<VKDevice> create_device ();

    VkDevice get_dev_id () const {
        return _dev_id;
    }
    SmartPtr<VkAllocationCallbacks>
    get_allocation_cb () const {
        return _allocator;
    }

    SmartPtr<VKShader> create_shader (const char *file_name);
    SmartPtr<VKShader> create_shader (const std::vector<uint32_t> &binary);
    //SmartPtr<VKPipeline> create_pipeline (const SmartPtr<VKShader> shader);
    SmartPtr<VKFence> create_fence (VkFenceCreateFlags flags = VK_FENCE_CREATE_SIGNALED_BIT);
    XCamReturn compute_queue_submit (const SmartPtr<VKCmdBuf> cmd_buf, const SmartPtr<VKFence> fence);
    XCamReturn compute_queue_wait_idle ();

protected:
    void destroy_shader_id (VkShaderModule shader);
    VkDeviceMemory allocate_mem_id (VkDeviceSize size, VkMemoryPropertyFlags memory_prop);
    void free_mem_id (VkDeviceMemory mem);
    XCamReturn map_mem (VkDeviceMemory mem, VkDeviceSize size, VkDeviceSize offset, void *&ptr);
    void unmap_mem (VkDeviceMemory mem);
    VkBuffer create_buf_id (VkBufferUsageFlags usage, uint32_t size);
    void destroy_buf_id (VkBuffer buf);
    XCamReturn bind_buffer (VkBuffer buf, VkDeviceMemory mem, VkDeviceSize offset = 0);

    VkDescriptorPool create_desc_pool (const VkDescriptorPoolCreateInfo &info);
    void destroy_desc_pool (VkDescriptorPool pool);

    VkDescriptorSet allocate_desc_set (const VkDescriptorSetAllocateInfo &info);
    XCamReturn free_desc_set (VkDescriptorSet set, VkDescriptorPool pool);

    XCamReturn update_desc_set (const std::vector<VkWriteDescriptorSet> &sets);
    VkCommandPool create_cmd_pool (VkFlags queue_flag = VK_QUEUE_COMPUTE_BIT);
    void destroy_cmd_pool (VkCommandPool pool);

    VkCommandBuffer allocate_cmd_buffer (VkCommandPool pool);
    void free_cmd_buffer (VkCommandPool pool, VkCommandBuffer buf);

    void destroy_fence (VkFence fence);
    XCamReturn reset_fence (VkFence fence);
    XCamReturn wait_for_fence (VkFence fence, uint64_t timeout);

protected:
    explicit VKDevice (VkDevice id, const SmartPtr<VKInstance> &instance);
    XCamReturn prepare_compute_queue ();
    //SmartPtr<VKLayout> create_desc_set_layout ();

private:
    XCAM_DEAD_COPY (VKDevice);

private:
    static SmartPtr<VKDevice>        _default_dev;
    static Mutex                     _default_mutex;

    VkDevice                         _dev_id;
    VkQueue                          _compute_queue;
    SmartPtr<VkAllocationCallbacks>  _allocator;
    SmartPtr<VKInstance>             _instance;
};

}

#endif  //XCAM_VK_DEVICE_H
