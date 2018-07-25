/*
 * vk_video_buf_allocator.h - vulkan video buffer allocator class
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

#ifndef XCAM_VK_VIDEO_BUF_ALLOCATOR_H
#define XCAM_VK_VIDEO_BUF_ALLOCATOR_H

#include <buffer_pool.h>

namespace XCam {

class VKDevice;
class VKBuffer;

class VKVideoBufAllocator
    : public BufferPool
{
public:
    explicit VKVideoBufAllocator (const SmartPtr<VKDevice> dev);
    virtual ~VKVideoBufAllocator ();

private:
    //derive from BufferPool
    virtual SmartPtr<BufferData> allocate_data (const VideoBufferInfo &buffer_info);
    virtual SmartPtr<BufferProxy> create_buffer_from_data (SmartPtr<BufferData> &data);

private:
    SmartPtr<VKDevice>     _dev;
};

class VKVideoBuffer
    : public BufferProxy
{
public:
    explicit VKVideoBuffer (const VideoBufferInfo &info, const SmartPtr<BufferData> &data);
    SmartPtr<VKBuffer> get_vk_buf ();
};

}

#endif //XCAM_VK_VIDEO_BUF_ALLOCATOR_H

