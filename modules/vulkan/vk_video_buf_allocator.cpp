/*
 * vk_video_buf_allocator.cpp - vulkan video buffer allocator implementation
 *
 *  Copyright (c) 2017 Intel Corporation
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

#include "vk_video_buf_allocator.h"
#include "vk_memory.h"
#include "vk_device.h"

namespace XCam {

class VKVideoData
    : public BufferData
{
    friend class VKVideoBuffer;
public:
    explicit VKVideoData (const SmartPtr<VKBuffer> vk_buf);
    virtual ~VKVideoData ();

    //derive from BufferData
    virtual uint8_t *map ();
    virtual bool unmap ();

    bool is_valid ();

private:
    uint8_t            *_mem_ptr;
    SmartPtr<VKBuffer>  _vk_buf;
};

VKVideoData::VKVideoData (const SmartPtr<VKBuffer> vk_buf)
    : _mem_ptr (NULL)
    , _vk_buf (vk_buf)
{
    XCAM_ASSERT (vk_buf.ptr ());
}

VKVideoData::~VKVideoData ()
{
}

bool
VKVideoData::is_valid ()
{
    return _vk_buf.ptr () && XCAM_IS_VALID_VK_ID (_vk_buf->get_buf_id ());
}

uint8_t *
VKVideoData::map ()
{
    if (!_mem_ptr) {
        _mem_ptr = (uint8_t *)_vk_buf->map ();
    }
    return _mem_ptr;
}

bool
VKVideoData::unmap ()
{
    if (!_mem_ptr)
        return false;

    _mem_ptr = NULL;
    _vk_buf->unmap ();
    return true;
}

VKVideoBufAllocator::VKVideoBufAllocator (const SmartPtr<VKDevice> dev)
    : _dev (dev)
{
}

VKVideoBufAllocator::~VKVideoBufAllocator ()
{
}

SmartPtr<BufferData>
VKVideoBufAllocator::allocate_data (const VideoBufferInfo &buffer_info)
{
    XCAM_FAIL_RETURN (
        ERROR, buffer_info.size, NULL,
        "VKVideoBufAllocator allocate data failed. buf_size is zero");

    SmartPtr<VKBuffer> vk_buf =
        VKBuffer::create_buffer (_dev, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, buffer_info.size);

    XCAM_FAIL_RETURN (
        ERROR, vk_buf.ptr (), NULL,
        "VKVideoBufAllocator create vk memory failed. buf_size :%d", buffer_info.size);

    VKBufInfo info;
    info.format = buffer_info.format;
    info.width = buffer_info.width;
    info.height = buffer_info.height;
    info.aligned_width = buffer_info.aligned_width;
    info.aligned_height = buffer_info.aligned_height;
    info.size = buffer_info.size;
    info.strides[0] = buffer_info.strides[0];
    info.strides[1] = buffer_info.strides[1];
    info.offsets[0] = buffer_info.offsets[0];
    info.offsets[1] = buffer_info.offsets[1];
    info.slice_size[0] = buffer_info.strides[0] * buffer_info.aligned_height;
    info.slice_size[1] = buffer_info.size - buffer_info.offsets[1];
    vk_buf->set_buf_info (info);

    SmartPtr<VKVideoData> data = new VKVideoData (vk_buf);
    XCAM_FAIL_RETURN (
        ERROR, data.ptr () && data->is_valid (), NULL,
        "VKVideoBufAllocator allocate data failed. buf_size:%d", buffer_info.size);

    return data;
}

SmartPtr<BufferProxy>
VKVideoBufAllocator::create_buffer_from_data (SmartPtr<BufferData> &data)
{
    const VideoBufferInfo &info = get_video_info ();

    XCAM_ASSERT (data.ptr ());
    return new VKVideoBuffer (info, data);
}

VKVideoBuffer::VKVideoBuffer (const VideoBufferInfo &info, const SmartPtr<BufferData> &data)
    : BufferProxy (info, data)
{
}

SmartPtr<VKBuffer>
VKVideoBuffer::get_vk_buf ()
{
    SmartPtr<BufferData> data = get_buffer_data ();
    SmartPtr<VKVideoData> vk_data = data.dynamic_cast_ptr<VKVideoData> ();
    XCAM_FAIL_RETURN (
        ERROR, vk_data.ptr () && vk_data->_vk_buf.ptr (), VK_NULL_HANDLE,
        "VKVideoBuffer get buf_id failed, data is empty");

    return vk_data->_vk_buf;
}

SmartPtr<BufferPool>
create_vk_buffer_pool (const SmartPtr<VKDevice> &dev)
{
    XCAM_FAIL_RETURN (
        ERROR, dev.ptr () && XCAM_IS_VALID_VK_ID(dev->get_dev_id()), NULL,
        "create_vk_buffer_pool failed since vk device is invalid");
    return new VKVideoBufAllocator (dev);
}

}
