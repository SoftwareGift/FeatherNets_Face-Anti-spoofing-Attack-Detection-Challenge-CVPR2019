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

#include "ocl/cl_context.h"
#include "ocl/cl_event.h"
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
    explicit CLMemory (const SmartPtr<CLContext> &context);
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

    XCamReturn enqueue_unmap (
        void *ptr,
        CLEventList &events_wait = CLEvent::EmptyList,
        SmartPtr<CLEvent> &event_out = CLEvent::NullEvent);

protected:
    void set_mem_id (cl_mem &id, bool need_destroy = true) {
        _mem_id = id;
        _mem_need_destroy = need_destroy;
    }

    void set_mapped_ptr (void *ptr) {
        _mapped_ptr = ptr;
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
    void                 *_mapped_ptr;
};

class CLBuffer
    : public CLMemory
{
protected:
    explicit CLBuffer (const SmartPtr<CLContext> &context);

public:
    explicit CLBuffer (
        const SmartPtr<CLContext> &context, uint32_t size,
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
    XCamReturn enqueue_map (
        void *&ptr, uint32_t offset, uint32_t size,
        cl_map_flags map_flags = CL_MEM_READ_WRITE,
        CLEventList &event_waits = CLEvent::EmptyList,
        SmartPtr<CLEvent> &event_out = CLEvent::NullEvent);

    uint32_t get_buf_size () {
        return _size;
    }

private:
    bool init_buffer (
        const SmartPtr<CLContext> &context, uint32_t size,
        cl_mem_flags  flags, void *host_ptr);

    XCAM_DEAD_COPY (CLBuffer);

private:
    cl_mem_flags    _flags;
    uint32_t        _size;
};

class CLVaBuffer
    : public CLBuffer
{
public:
    explicit CLVaBuffer (
        const SmartPtr<CLContext> &context,
        SmartPtr<DrmBoBuffer> &bo);

private:
    bool init_va_buffer (const SmartPtr<CLContext> &context, SmartPtr<DrmBoBuffer> &bo);

    XCAM_DEAD_COPY (CLVaBuffer);

private:
    SmartPtr<DrmBoBuffer>   _bo;
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

    XCamReturn enqueue_map (
        void *&ptr,
        size_t *origin, size_t *region,
        size_t *row_pitch, size_t *slice_pitch,
        cl_map_flags map_flags = CL_MEM_READ_WRITE,
        CLEventList &event_waits = CLEvent::EmptyList,
        SmartPtr<CLEvent> &event_out = CLEvent::NullEvent);

protected:
    explicit CLImage (const SmartPtr<CLContext> &context);
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
        const SmartPtr<CLContext> &context,
        SmartPtr<DrmBoBuffer> &bo,
        uint32_t offset = 0,
        bool single_plane = false);
    explicit CLVaImage (
        const SmartPtr<CLContext> &context,
        SmartPtr<DrmBoBuffer> &bo,
        const CLImageDesc &image_info,
        uint32_t offset = 0);
    ~CLVaImage () {}

private:
    bool init_va_image (
        const SmartPtr<CLContext> &context, SmartPtr<DrmBoBuffer> &bo,
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
        const SmartPtr<CLContext> &context,
        const VideoBufferInfo &video_info,
        cl_mem_flags  flags = CL_MEM_READ_WRITE);

    explicit CLImage2D (
        const SmartPtr<CLContext> &context,
        const CLImageDesc &cl_desc,
        cl_mem_flags  flags = CL_MEM_READ_WRITE,
        SmartPtr<CLBuffer> bind_buf = NULL);

    SmartPtr<CLBuffer> get_bind_buf () {
        return _bind_buf;
    }

    ~CLImage2D () {}

private:
    bool init_image_2d (
        const SmartPtr<CLContext> &context,
        const CLImageDesc &cl_desc,
        cl_mem_flags  flags);

    XCAM_DEAD_COPY (CLImage2D);

private:
    SmartPtr<CLBuffer> _bind_buf;
};

class CLImage2DArray
    : public CLImage
{
public:
    explicit CLImage2DArray (
        const SmartPtr<CLContext> &context,
        const VideoBufferInfo &video_info,
        cl_mem_flags  flags = CL_MEM_READ_WRITE,
        uint32_t extra_array_size = 0);

    ~CLImage2DArray () {}

private:
    bool init_image_2d_array (
        const SmartPtr<CLContext> &context,
        const CLImageDesc &cl_desc,
        cl_mem_flags  flags);

    XCAM_DEAD_COPY (CLImage2DArray);
};


};
#endif //
