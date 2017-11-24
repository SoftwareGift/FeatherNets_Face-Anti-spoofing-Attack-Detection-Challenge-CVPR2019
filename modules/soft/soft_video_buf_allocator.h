/*
 * soft_video_buf_allocator.h - soft video buffer allocator class
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

#ifndef XCAM_SOFT_VIDEO_BUF_ALLOCATOR_H
#define XCAM_SOFT_VIDEO_BUF_ALLOCATOR_H

#include <xcam_std.h>
#include <buffer_pool.h>

namespace XCam {

class SoftVideoBufAllocator
    : public BufferPool
{
public:
    explicit SoftVideoBufAllocator ();
    explicit SoftVideoBufAllocator (const VideoBufferInfo &info);
    virtual ~SoftVideoBufAllocator ();

private:
    //derive from BufferPool
    virtual SmartPtr<BufferData> allocate_data (const VideoBufferInfo &buffer_info);
};

#if 0
class AllocatorPool {
public:
    explicit AllocatorPool ();
    virtual ~AllocatorPool ();

    SmartPtr<VideoBuffer> allocate_video_buf (const VideoBufferInfo &info);

private:
    SafeList<BufferPool> _pools;
};
#endif

}

#endif //XCAM_SOFT_VIDEO_BUF_ALLOCATOR_H
