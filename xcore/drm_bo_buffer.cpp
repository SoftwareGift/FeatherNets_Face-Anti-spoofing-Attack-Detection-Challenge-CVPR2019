/*
 * drm_bo_buffer.cpp - drm bo buffer
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

#include "drm_bo_buffer.h"

namespace XCam {

DrmBoWrapper::DrmBoWrapper (SmartPtr<DrmDisplay> &display, drm_intel_bo *bo)
    : _display (display)
    , _bo (bo)
{
    XCAM_ASSERT (display.ptr());
    XCAM_ASSERT (bo);
}

DrmBoWrapper::~DrmBoWrapper ()
{
    if (_bo)
        drm_intel_bo_unreference (_bo);
}

DrmBoBuffer::DrmBoBuffer (
    SmartPtr<DrmDisplay> display,
    const VideoBufferInfo &info,
    SmartPtr<DrmBoWrapper> &bo)
    : VideoBuffer (info)
    , _display (display)
    , _bo (bo)
{
    XCAM_ASSERT (bo.ptr ());
}

DrmBoBuffer::~DrmBoBuffer ()
{
    if (_pool.ptr ()) {
        _pool->release (_bo);
    } else if (_bo.ptr())
        _bo.release ();
    _parent.release ();
    _pool.release ();
}

void
DrmBoBuffer::set_parent (SmartPtr<VideoBuffer> &parent)
{
    _parent = parent;
}

void
DrmBoBuffer::set_buf_pool (SmartPtr<DrmBoBufferPool> &buf_pool)
{
    _pool = buf_pool;
}

uint8_t *
DrmBoBuffer::map ()
{
    return NULL;
}

bool DrmBoBuffer::unmap ()
{
    return true;
}

DrmBoBufferPool::DrmBoBufferPool (SmartPtr<DrmDisplay> &display)
    : _display (display)
    , _buf_count (0)
{
    XCAM_ASSERT (display.ptr ());
    XCAM_LOG_DEBUG ("DrmBoBufferPool constructed");
}

DrmBoBufferPool::~DrmBoBufferPool ()
{
    deinit ();
    _display.release ();
    XCAM_LOG_DEBUG ("DrmBoBufferPool destructed");
}

bool
DrmBoBufferPool::set_buffer_info (const VideoBufferInfo &info)
{
    XCAM_ASSERT (info.format && info.width && info.height);
    _buf_info = info;
    return true;
}

bool
DrmBoBufferPool::init (uint32_t buf_num)
{
    uint32_t i = 0;

    XCAM_ASSERT (_buf_info.format);
    XCAM_ASSERT (buf_num > 0);

    for (i = 0; i < buf_num; ++i) {
        SmartPtr<DrmBoWrapper> bo = _display->create_drm_bo (_display, _buf_info);
        if (!bo.ptr())
            break;
        _buf_list.push (bo);
    }
    if (i == 0) {
        XCAM_LOG_ERROR ("DrmBoBufferPool failed to allocate %d buffers", buf_num);
        return false;
    }
    if (i != buf_num) {
        XCAM_LOG_WARNING (
            "DrmBoBufferPool expect for %d buf but only allocate %d buf",
            buf_num, i);
    }
    _buf_count = i;
    return true;
}

void
DrmBoBufferPool::deinit ()
{
    _buf_list.wakeup ();
    _buf_list.clear ();
}

void
DrmBoBufferPool::release (SmartPtr<DrmBoWrapper> &bo)
{
    _buf_list.push (bo);
}

SmartPtr<DrmBoBuffer>
DrmBoBufferPool::get_buffer (SmartPtr<DrmBoBufferPool> &self)
{
    SmartPtr<DrmBoBuffer> bo_buf;
    SmartPtr<DrmBoWrapper> bo;

    XCAM_ASSERT (self.ptr () == this);
    XCAM_FAIL_RETURN(
        WARNING,
        self.ptr () == this,
        NULL,
        "DrmBoBufferPool get_buffer failed since parameter<self> not this");

    bo = _buf_list.pop ();
    if (!bo.ptr ()) {
        XCAM_LOG_DEBUG ("DrmBoBufferPool failed to get buffer");
        return NULL;
    }
    bo_buf =  new DrmBoBuffer (_display, _buf_info, bo);
    bo_buf->set_buf_pool (self);
    return bo_buf;
}

};
