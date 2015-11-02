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
#include "x3a_stats_pool.h"

#define OCL_TILING_NONE    0

namespace XCam {

DrmBoData::DrmBoData (SmartPtr<DrmDisplay> &display, drm_intel_bo *bo)
    : _display (display)
    , _bo (bo)
    , _buf (NULL)
    , _prime_fd (-1)
{
    XCAM_ASSERT (display.ptr ());
    XCAM_ASSERT (bo);
}

DrmBoData::~DrmBoData ()
{
    unmap ();
    if (_bo)
        drm_intel_bo_unreference (_bo);
}

uint8_t *
DrmBoData::map ()
{
    if (_buf) {
        return _buf;
    }

    uint32_t tiling_mode, swizzle_mode;

    drm_intel_bo_get_tiling (_bo, &tiling_mode, &swizzle_mode);

    if (tiling_mode != OCL_TILING_NONE) {
        if (drm_intel_gem_bo_map_gtt (_bo) != 0)
            return NULL;
    }
    else {
        if (drm_intel_bo_map (_bo, 1) != 0)
            return NULL;
    }

    _buf = (uint8_t *)_bo->virt;
    return  _buf;
}

bool
DrmBoData::unmap ()
{
    if (!_buf || !_bo)
        return true;

    uint32_t tiling_mode, swizzle_mode;

    drm_intel_bo_get_tiling (_bo, &tiling_mode, &swizzle_mode);

    if (tiling_mode != OCL_TILING_NONE) {
        if (drm_intel_gem_bo_unmap_gtt (_bo) != 0)
            return false;
    }
    else {
        if (drm_intel_bo_unmap (_bo) != 0)
            return false;
    }

    _buf = NULL;
    return true;
}

int
DrmBoData::get_fd ()
{
    if (_prime_fd == -1) {
        if (drm_intel_bo_gem_export_to_prime (_bo, &_prime_fd) < 0) {
            _prime_fd = -1;
            XCAM_LOG_DEBUG ("DrmBoData: failed to obtain prime fd");
        }
    }

    return _prime_fd;
}

DrmBoBuffer::DrmBoBuffer (const VideoBufferInfo &info, const SmartPtr<DrmBoData> &data)
    : BufferProxy (info, data)
{
    XCAM_ASSERT (data.ptr ());
}

drm_intel_bo *
DrmBoBuffer::get_bo ()
{
    SmartPtr<BufferData> data = get_buffer_data ();
    SmartPtr<DrmBoData> bo = data.dynamic_cast_ptr<DrmBoData> ();

    XCAM_FAIL_RETURN(
        WARNING,
        bo.ptr(),
        NULL,
        "DrmBoBuffer get_buffer_data failed with NULL");
    return bo->get_bo ();
}

SmartPtr<X3aStats>
DrmBoBuffer::find_3a_stats ()
{
    for (VideoBufferList::iterator iter = _attached_bufs.begin ();
            iter != _attached_bufs.end (); ++iter) {
        SmartPtr<X3aStats> stats = (*iter).dynamic_cast_ptr<X3aStats> ();
        if (stats.ptr ())
            return stats;
    }

    return NULL;
}

DrmBoBufferPool::DrmBoBufferPool (SmartPtr<DrmDisplay> &display)
    : _display (display)
{
    XCAM_ASSERT (display.ptr ());
    XCAM_LOG_DEBUG ("DrmBoBufferPool constructed");
}

DrmBoBufferPool::~DrmBoBufferPool ()
{
    _display.release ();
    XCAM_LOG_DEBUG ("DrmBoBufferPool destructed");
}

SmartPtr<BufferData>
DrmBoBufferPool::allocate_data (const VideoBufferInfo &buffer_info)
{
    SmartPtr<DrmBoData> bo = _display->create_drm_bo (_display, buffer_info);
    return bo;
}

SmartPtr<BufferProxy>
DrmBoBufferPool::create_buffer_from_data (SmartPtr<BufferData> &data)
{
    const VideoBufferInfo & info = get_video_info ();
    SmartPtr<DrmBoData> bo_data = data.dynamic_cast_ptr<DrmBoData> ();
    XCAM_ASSERT (bo_data.ptr ());

    return new DrmBoBuffer (info, bo_data);
}

};
