/*
 * vk_memory.cpp - Vulkan memory
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

#include "vk_memory.h"
#include "vk_device.h"

namespace XCam {

VKBufInfo::VKBufInfo ()
    : format (V4L2_PIX_FMT_NV12)
    , width (0)
    , height (0)
    , aligned_width (0)
    , aligned_height (0)
    , size (0)
{
    xcam_mem_clear (strides);
    xcam_mem_clear (offsets);
    xcam_mem_clear (slice_size);
}

VKMemory::VKMemory (
    const SmartPtr<VKDevice> dev,
    VkDeviceMemory id,
    uint32_t size,
    VkMemoryPropertyFlags mem_prop)
    : _dev (dev)
    , _mem_id (id)
    , _mem_prop (mem_prop)
    , _size (size)
    , _mapped_ptr (NULL)
{
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (id));
}

VKMemory::~VKMemory ()
{
    if (XCAM_IS_VALID_VK_ID (_mem_id) && _dev.ptr ()) {
        _dev->free_mem_id (_mem_id);
    }
}

void *
VKMemory::map (VkDeviceSize size, VkDeviceSize offset)
{
    if (_mapped_ptr)
        return _mapped_ptr;

    XCAM_FAIL_RETURN (
        ERROR,
        xcam_ret_is_ok (_dev->map_mem (_mem_id, size, offset, _mapped_ptr)), NULL,
        "VK memory map failed");

    return _mapped_ptr;
}

void
VKMemory::unmap ()
{
    if (_mapped_ptr) {
        _dev->unmap_mem (_mem_id);
        _mapped_ptr = NULL;
    }
}

VKBuffer::VKBuffer (
    const SmartPtr<VKDevice> dev,
    VkBuffer buf_id,
    VkDeviceMemory mem_id,
    uint32_t size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags prop)
    : VKMemory (dev, mem_id, size, prop)
    , _buffer_id (buf_id)
    , _usage_flags (usage)
    , _prop_flags (prop)
{
}

VKBuffer::~VKBuffer ()
{
    if (XCAM_IS_VALID_VK_ID (_buffer_id) && _dev.ptr ()) {
        _dev->destroy_buf_id (_buffer_id);
    }
}

XCamReturn
VKBuffer::bind ()
{
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (_buffer_id));
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (_mem_id));

    return _dev->bind_buffer (_buffer_id, _mem_id, 0);
}

SmartPtr<VKBuffer>
VKBuffer::create_buffer (
    const SmartPtr<VKDevice> dev,
    VkBufferUsageFlags usage,
    uint32_t size,  void *data,
    VkMemoryPropertyFlags mem_prop)
{
    XCAM_FAIL_RETURN (
        ERROR, dev.ptr () && size, NULL,
        "vk create buffer failed because of dev or size errors");

    VkBuffer buf_id = dev->create_buf_id (usage, size);
    XCAM_FAIL_RETURN (
        ERROR, XCAM_IS_VALID_VK_ID (buf_id), NULL,
        "vk create buffer failed");

    VkDevice dev_id = dev->get_dev_id ();
    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements (dev_id, buf_id, &mem_reqs);
    VkDeviceMemory mem_id = dev->allocate_mem_id (mem_reqs.size, mem_prop);
    XCAM_FAIL_RETURN (
        ERROR, XCAM_IS_VALID_VK_ID (mem_id), NULL,
        "vk create buffer failed in mem allocation");

    // size == mem_reqs.size or size?
    SmartPtr<VKBuffer> buf = new VKBuffer (dev, buf_id, mem_id, size, usage, mem_prop);

    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (buf->bind ()), NULL,
        "vk create bufer failed when bind with memory");
    if (!data)
        return buf;

    void *ptr = buf->map ();
    XCAM_FAIL_RETURN (
        ERROR, ptr, NULL,
        "vk create bufer failed when map the buf");
    memcpy (ptr, data, size);
    buf->unmap ();

    return buf;

}

VKBufDesc::VKBufDesc ()
{
    xcam_mem_clear (desc_info);
}

VKBufDesc::VKBufDesc (const SmartPtr<VKBuffer> &buffer, NV12PlaneIdx plane)
    : buf (buffer)
{
    xcam_mem_clear (desc_info);
    const VKBufInfo info = buffer->get_buf_info ();

    desc_info.buffer = buffer->get_buf_id ();
    desc_info.offset = info.offsets[plane];
    desc_info.range = info.slice_size[plane];
}

VKBufDesc::VKBufDesc (const SmartPtr<VKBuffer> &buffer, uint32_t offset, size_t size)
    : buf (buffer)
{
    xcam_mem_clear (desc_info);
    desc_info.buffer = buffer->get_buf_id ();
    desc_info.offset = offset;
    desc_info.range = size;
}

}
