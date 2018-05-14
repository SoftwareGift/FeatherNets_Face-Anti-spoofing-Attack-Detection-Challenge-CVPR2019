/*
 * vk_memory.h - Vulkan memory
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

#ifndef XCAM_VK_MEMORY_H
#define XCAM_VK_MEMORY_H

#include <vulkan/vulkan_std.h>

namespace XCam {

class VKDevice;

class VKMemory
{
public:
    virtual ~VKMemory ();
    void *map (VkDeviceSize size = VK_WHOLE_SIZE, VkDeviceSize offset = 0);
    void unmap ();

protected:
    explicit VKMemory (
        const SmartPtr<VKDevice> dev, VkDeviceMemory id,
        uint32_t size, VkMemoryPropertyFlags mem_prop);
    VkDeviceMemory get_mem_id () const {
        return _mem_id;
    }

private:
    XCAM_DEAD_COPY (VKMemory);

protected:
    const SmartPtr<VKDevice>     _dev;
    VkDeviceMemory               _mem_id;
    VkMemoryPropertyFlags        _mem_prop;
    uint32_t                     _size;
    void                        *_mapped_ptr;
};

class VKBuffer
    : public VKMemory
{
public:
    ~VKBuffer ();

    // usage can be VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
    static SmartPtr<VKBuffer>
    create_buffer (
        const SmartPtr<VKDevice> dev,
        VkBufferUsageFlags usage,
        uint32_t size,  void *data = NULL,
        VkMemoryPropertyFlags mem_prop = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkBuffer get_buf_id () const {
        return _buffer_id;
    }
    VkBufferUsageFlags get_usage_flags () const {
        return _usage_flags;
    }
    VkMemoryPropertyFlags get_mem_flags () const {
        return _prop_flags;
    }

private:
    explicit VKBuffer (
        const SmartPtr<VKDevice> dev, VkBuffer buf_id,
        VkDeviceMemory mem_id, uint32_t size,
        VkBufferUsageFlags usage, VkMemoryPropertyFlags prop);
    XCamReturn bind ();

private:
    XCAM_DEAD_COPY (VKBuffer);

private:
    VkBuffer                         _buffer_id;
    VkBufferUsageFlags               _usage_flags;
    VkMemoryPropertyFlags            _prop_flags;
};

struct VKBufDesc {
    SmartPtr<VKBuffer>        buf;
    VkDescriptorBufferInfo    desc_info;

    VKBufDesc ();
    VKBufDesc (const SmartPtr<VKBuffer> &buffer, uint32_t offset = 0, size_t size = VK_WHOLE_SIZE);
};

typedef std::vector<SmartPtr<VKBuffer>>  VKBufferArray;

}

#endif  //XCAM_VK_MEMORY_H
