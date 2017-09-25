/*
 * cl_video_buffer.cpp - cl video buffer
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
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#include "ocl/cl_memory.h"
#include "ocl/cl_device.h"
#include "ocl/cl_video_buffer.h"

namespace XCam {

CLVideoBufferData::CLVideoBufferData (SmartPtr<CLBuffer> &body)
    : _buf_ptr (NULL)
    , _buf (body)
{
    XCAM_ASSERT (body.ptr ());
}

CLVideoBufferData::~CLVideoBufferData ()
{
    unmap ();
    _buf.release ();
}

cl_mem &
CLVideoBufferData::get_mem_id () {
    return _buf->get_mem_id ();
}

uint8_t *
CLVideoBufferData::map ()
{
    if (_buf_ptr)
        return _buf_ptr;

    uint32_t size = _buf->get_buf_size ();
    XCamReturn ret = _buf->enqueue_map ((void*&) _buf_ptr, 0, size);
    XCAM_FAIL_RETURN (
        ERROR,
        ret == XCAM_RETURN_NO_ERROR,
        NULL,
        "CLVideoBufferData map data failed");

    return _buf_ptr;
}

bool
CLVideoBufferData::unmap ()
{
    if (!_buf_ptr)
        return true;

    XCamReturn ret = _buf->enqueue_unmap ((void*&) _buf_ptr);
    XCAM_FAIL_RETURN (
        ERROR,
        ret == XCAM_RETURN_NO_ERROR,
        NULL,
        "CLVideoBufferData unmap data failed");

    _buf_ptr = NULL;
    return true;
}

CLVideoBuffer::CLVideoBuffer (
    const SmartPtr<CLContext> &context, const VideoBufferInfo &info, const SmartPtr<CLVideoBufferData> &data)
    : BufferProxy (info, data)
    , CLBuffer (context)
{
    XCAM_ASSERT (data.ptr ());

    SmartPtr<CLBuffer> cl_buf = data->get_cl_buffer ();
    XCAM_ASSERT (cl_buf.ptr ());
    set_mem_id (cl_buf->get_mem_id (), false);
    set_buf_size (cl_buf->get_buf_size ());
}

SmartPtr<CLBuffer>
CLVideoBuffer::get_cl_buffer ()
{
    SmartPtr<BufferData> data = get_buffer_data ();
    SmartPtr<CLVideoBufferData> cl_data = data.dynamic_cast_ptr<CLVideoBufferData> ();
    XCAM_FAIL_RETURN(
        WARNING,
        cl_data.ptr(),
        NULL,
        "CLVideoBuffer get buffer data failed with NULL");

    return cl_data->get_cl_buffer ();
}

SmartPtr<X3aStats>
CLVideoBuffer::find_3a_stats ()
{
    return find_typed_attach<X3aStats> ();
}

bool
CLVideoBufferPool::fixate_video_info (VideoBufferInfo &info)
{
    if (info.format != V4L2_PIX_FMT_NV12)
        return true;

    VideoBufferInfo out_info;
    out_info.init (info.format, info.width, info.height, info.aligned_width, info.aligned_height);

    return true;
}

SmartPtr<BufferData>
CLVideoBufferPool::allocate_data (const VideoBufferInfo &buffer_info)
{
    SmartPtr<CLContext> context = CLDevice::instance ()->get_context ();

    SmartPtr<CLBuffer> buf = new CLBuffer (context, buffer_info.size);
    XCAM_ASSERT (buf.ptr ());

    return new CLVideoBufferData (buf);
}

SmartPtr<BufferProxy>
CLVideoBufferPool::create_buffer_from_data (SmartPtr<BufferData> &data)
{
    SmartPtr<CLContext> context = CLDevice::instance ()->get_context ();
    const VideoBufferInfo & info = get_video_info ();
    SmartPtr<CLVideoBufferData> cl_data = data.dynamic_cast_ptr<CLVideoBufferData> ();
    XCAM_ASSERT (cl_data.ptr ());

    SmartPtr<CLVideoBuffer> buf = new CLVideoBuffer (context, info, cl_data);
    XCAM_ASSERT (buf.ptr ());

    return buf;
}

};
