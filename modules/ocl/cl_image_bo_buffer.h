/*
 * cl_image_bo_buffer.h - cl image bo buffer
 *
 *  Copyright (c) 2015 Intel Corporation
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

#ifndef XCAM_CL_IMAGE_BO_BUFFER_H
#define XCAM_CL_IMAGE_BO_BUFFER_H

#include <xcam_std.h>
#include <drm_bo_buffer.h>
#include <ocl/cl_memory.h>
#include <ocl/cl_context.h>

namespace XCam {

class CLImageBoBuffer;
class CLBoBufferPool;

class CLImageBoData
    : public DrmBoData
{
    friend class CLBoBufferPool;
    friend class CLImageBoBuffer;
public:
    virtual int get_fd ();

private:
    explicit CLImageBoData (SmartPtr<DrmDisplay> &display, SmartPtr<CLImage> &image, drm_intel_bo *bo);
    XCAM_DEAD_COPY (CLImageBoData);

    SmartPtr<CLImage> &get_image () {
        return _image;
    }

private:
    SmartPtr<CLImage>          _image;
};

class CLImageBoBuffer
    : public DrmBoBuffer
{
    friend class CLBoBufferPool;
public:
    SmartPtr<CLImage> get_cl_image ();

protected:
    CLImageBoBuffer (const VideoBufferInfo &info, const SmartPtr<CLImageBoData> &data);

    //derived from SwappedBuffer
    virtual SmartPtr<SwappedBuffer> create_new_swap_buffer (
        const VideoBufferInfo &info, SmartPtr<BufferData> &data);
};

class CLBoBufferPool
    : public DrmBoBufferPool
{
public:
    explicit CLBoBufferPool (SmartPtr<DrmDisplay> &display, SmartPtr<CLContext> &context);
    ~CLBoBufferPool ();

protected:
    // derived from BufferPool
    virtual bool fixate_video_info (VideoBufferInfo &info);
    virtual SmartPtr<BufferData> allocate_data (const VideoBufferInfo &buffer_info);
    virtual SmartPtr<BufferProxy> create_buffer_from_data (SmartPtr<BufferData> &data);

private:
    SmartPtr<CLImageBoData> create_image_bo (const VideoBufferInfo &buffer_info);
    XCAM_DEAD_COPY (CLBoBufferPool);

private:
    SmartPtr<CLContext>     _context;
};


};
#endif // XCAM_CL_IMAGE_BO_BUFFER_H
