/*
 * cl_memory.h - CL memory
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

#ifndef XCAM_CL_MEMORY_H
#define XCAM_CL_MEMORY_H

#include "xcam_utils.h"
#include "cl_context.h"
#include "cl_event.h"
#include "drm_bo_buffer.h"

namespace XCam {

struct CLImageDesc {
    cl_image_format         format;
    uint32_t                width;
    uint32_t                height;
    uint32_t                row_pitch;
    uint32_t                slice_pitch;
    uint32_t                array_size;
    uint32_t                size;

    CLImageDesc ();
    bool operator == (const CLImageDesc& desc) const;
};

class CLMemory {
public:
    explicit CLMemory (SmartPtr<CLContext> &context);
    virtual ~CLMemory ();

    cl_mem &get_mem_id () {
        return _mem_id;
    }
    bool is_valid () const {
        return _mem_id != NULL;
    }

    bool get_cl_mem_info (
        cl_image_info param_name, size_t param_size,
        void *param, size_t *param_size_ret = NULL);

    int32_t export_fd ();
    void release_fd ();

protected:
    void set_mem_id (cl_mem &id, bool need_destroy = true) {
        _mem_id = id;
        _mem_need_destroy = need_destroy;
    }
    SmartPtr<CLContext> &get_context () {
        return _context;
    }

private:
    XCAM_DEAD_COPY (CLMemory);

private:
    SmartPtr<CLContext>   _context;
    cl_mem                _mem_id;
    int32_t               _mem_fd;
    bool                  _mem_need_destroy;
};

class CLBuffer
    : public CLMemory
{
public:
    explicit CLBuffer (
        SmartPtr<CLContext> &context, uint32_t size,
        cl_mem_flags  flags =  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        void *host_ptr = NULL);

    XCamReturn enqueue_read (
        void *ptr, uint32_t offset, uint32_t size,
        CLEventList &event_waits = CLEvent::EmptyList,
        SmartPtr<CLEvent> &event_out = CLEvent::NullEvent);
    XCamReturn enqueue_write (
        void *ptr, uint32_t offset, uint32_t size,
        CLEventList &event_waits = CLEvent::EmptyList,
        SmartPtr<CLEvent> &event_out = CLEvent::NullEvent);

private:
    bool init_buffer (
        SmartPtr<CLContext> &context, uint32_t size,
        cl_mem_flags  flags, void *host_ptr);

    XCAM_DEAD_COPY (CLBuffer);

private:
    cl_mem_flags    _flags;
    uint32_t        _size;
};

class CLImage
    : public CLMemory
{
public:
    virtual ~CLImage () {}

    const CLImageDesc &get_image_desc () const {
        return _image_desc;
    }
    uint32_t get_pixel_bytes () const;

    static uint32_t calculate_pixel_bytes (const cl_image_format &fmt);
    static bool video_info_2_cl_image_desc (
        const VideoBufferInfo & video_info,
        CLImageDesc &cl_desc);

protected:
    explicit CLImage (SmartPtr<CLContext> &context);
    void init_desc_by_image ();
    bool get_cl_image_info (
        cl_image_info param_name, size_t param_size,
        void *param, size_t *param_size_ret = NULL);

private:
    XCAM_DEAD_COPY (CLImage);

    CLImageDesc  _image_desc;
};

class CLVaImage
    : public CLImage
{
public:
    explicit CLVaImage (
        SmartPtr<CLContext> &context,
        SmartPtr<DrmBoBuffer> &bo,
        uint32_t offset = 0);
    explicit CLVaImage (
        SmartPtr<CLContext> &context,
        SmartPtr<DrmBoBuffer> &bo,
        const CLImageDesc &image_info,
        uint32_t offset = 0);
    ~CLVaImage () {}

private:
    bool init_va_image (
        SmartPtr<CLContext> &context, SmartPtr<DrmBoBuffer> &bo,
        const CLImageDesc &cl_desc, uint32_t offset);
    bool merge_multi_plane (
        const VideoBufferInfo &video_info,
        CLImageDesc &cl_desc);

    XCAM_DEAD_COPY (CLVaImage);

private:
    SmartPtr<DrmBoBuffer>   _bo;
    cl_libva_image          _va_image_info;
};

class CLImage2D
    : public CLImage
{
public:
    explicit CLImage2D (
        SmartPtr<CLContext> &context,
        const VideoBufferInfo &video_info,
        cl_mem_flags  flags = CL_MEM_READ_WRITE);

    ~CLImage2D () {}

private:
    bool init_image_2d (
        SmartPtr<CLContext> &context,
        const CLImageDesc &cl_desc,
        cl_mem_flags  flags);

    XCAM_DEAD_COPY (CLImage2D);
};

class CLImage2DArray
    : public CLImage
{
public:
    explicit CLImage2DArray (
        SmartPtr<CLContext> &context,
        const VideoBufferInfo &video_info,
        cl_mem_flags  flags = CL_MEM_READ_WRITE);

    ~CLImage2DArray () {}

private:
    bool init_image_2d_array (
        SmartPtr<CLContext> &context,
        const CLImageDesc &cl_desc,
        cl_mem_flags  flags);

    XCAM_DEAD_COPY (CLImage2DArray);
};


};
#endif //
