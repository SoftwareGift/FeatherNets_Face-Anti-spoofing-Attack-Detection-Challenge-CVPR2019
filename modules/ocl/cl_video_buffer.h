/*
 * cl_video_buffer.h - cl video buffer
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
#ifndef XCAM_CL_VIDEO_BUFFER_H
#define XCAM_CL_VIDEO_BUFFER_H

#include <xcam_std.h>
#include <safe_list.h>
#include <xcam_mutex.h>
#include <buffer_pool.h>
#include <x3a_stats_pool.h>
#include <ocl/cl_context.h>

namespace XCam {

class CLBuffer;
class CLVideoBufferPool;

class CLVideoBufferData
    : public BufferData
{
    friend class CLVideoBufferPool;

public:
    ~CLVideoBufferData ();

    cl_mem &get_mem_id ();
    SmartPtr<CLBuffer> get_cl_buffer () {
        return _buf;
    }

    //derived from BufferData
    virtual uint8_t *map ();
    virtual bool unmap ();

protected:
    explicit CLVideoBufferData (SmartPtr<CLBuffer> &body);

private:
    XCAM_DEAD_COPY (CLVideoBufferData);

private:
    uint8_t                *_buf_ptr;
    SmartPtr<CLBuffer>      _buf;
};

class CLVideoBuffer
    : public BufferProxy
    , public CLBuffer
{
    friend class CLVideoBufferPool;

public:
    explicit CLVideoBuffer (
        const SmartPtr<CLContext> &context, const VideoBufferInfo &info, const SmartPtr<CLVideoBufferData> &data);
    virtual ~CLVideoBuffer () {}

    SmartPtr<CLBuffer> get_cl_buffer ();
    SmartPtr<X3aStats> find_3a_stats ();

protected:
    CLVideoBuffer (const VideoBufferInfo &info, const SmartPtr<CLVideoBufferData> &data);

private:
    XCAM_DEAD_COPY (CLVideoBuffer);
};

class CLVideoBufferPool
    : public BufferPool
{
    friend class CLVideoBuffer;

public:
    explicit CLVideoBufferPool () {}
    ~CLVideoBufferPool () {}

protected:
    // derived from BufferPool
    virtual bool fixate_video_info (VideoBufferInfo &info);
    virtual SmartPtr<BufferData> allocate_data (const VideoBufferInfo &buffer_info);
    virtual SmartPtr<BufferProxy> create_buffer_from_data (SmartPtr<BufferData> &data);

private:
    XCAM_DEAD_COPY (CLVideoBufferPool);
};

};

#endif // XCAM_CL_VIDEO_BUFFER_H
