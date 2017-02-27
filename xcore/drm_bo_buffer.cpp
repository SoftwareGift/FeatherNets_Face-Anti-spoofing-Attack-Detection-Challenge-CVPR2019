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
    , _need_close_fd (true)
{
    XCAM_ASSERT (display.ptr ());
    XCAM_ASSERT (bo);
}

DrmBoData::~DrmBoData ()
{
    unmap ();
    if (_bo)
        drm_intel_bo_unreference (_bo);
    if (_prime_fd != -1 && _need_close_fd)
        close (_prime_fd);
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
            XCAM_LOG_ERROR ("DrmBoData: failed to obtain prime fd: %s", strerror(errno));
        }
        _need_close_fd = true;
    }

    return _prime_fd;
}

bool
DrmBoData::set_prime_fd (int fd, bool need_close)
{
    if (_prime_fd != -1) {
        XCAM_LOG_ERROR ("DrmBoData: set_dma_fd failed, the current prime fd was already set");
        return false;
    }
    _prime_fd = fd;
    _need_close_fd = need_close;
    return true;
}

DrmBoBuffer::DrmBoBuffer (const VideoBufferInfo &info, const SmartPtr<DrmBoData> &data)
    : BufferProxy (info, data)
    , SwappedBuffer (info, data)
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
    return find_typed_attach<X3aStats> ();
}

SmartPtr<SwappedBuffer>
DrmBoBuffer::create_new_swap_buffer (
    const VideoBufferInfo &info, SmartPtr<BufferData> &data)
{
    XCAM_ASSERT (get_buffer_data ().ptr () == data.ptr ());

    SmartPtr<DrmBoData> bo = data.dynamic_cast_ptr<DrmBoData> ();

    XCAM_FAIL_RETURN(
        WARNING,
        bo.ptr(),
        NULL,
        "DrmBoBuffer create_new_swap_buffer failed with NULL buffer data");

    return new DrmBoBuffer (info, bo);
}

DrmBoBufferPool::DrmBoBufferPool (SmartPtr<DrmDisplay> &display)
    : _swap_flags (SwappedBuffer::SwapNone)
    , _swap_init_order ((uint32_t)(SwappedBuffer::OrderY0Y1) | (uint32_t)(SwappedBuffer::OrderUV0UV1))
    , _display (display)
{
    xcam_mem_clear (_swap_offsets);
    XCAM_ASSERT (display.ptr ());
    XCAM_LOG_DEBUG ("DrmBoBufferPool constructed");
}

DrmBoBufferPool::~DrmBoBufferPool ()
{
    _display.release ();
    XCAM_LOG_DEBUG ("DrmBoBufferPool destructed");
}

bool
DrmBoBufferPool::update_swap_init_order (uint32_t init_order)
{
    VideoBufferInfo info = get_video_info ();
    XCAM_ASSERT (info.format);

    if ((_swap_flags & (uint32_t)(SwappedBuffer::SwapY)) && !(init_order & (uint32_t)(SwappedBuffer::OrderYMask))) {
        XCAM_LOG_WARNING ("update swap init order failed, need init Y order, error order:0x%04x", init_order);
        return false;
    }

    if ((_swap_flags & (uint32_t)(SwappedBuffer::SwapUV)) && !(init_order & (uint32_t)(SwappedBuffer::OrderUVMask))) {
        XCAM_LOG_WARNING ("update swap init order failed, need init UV order, error order:0x%04x", init_order);
        return false;
    }
    _swap_init_order = init_order;

    XCAM_FAIL_RETURN (
        WARNING,
        init_swap_order (info),
        false,
        "CL3aImageProcessor post_config failed");

    update_video_info_unsafe (info);

    return true;
}

bool
DrmBoBufferPool::fixate_video_info (VideoBufferInfo &info)
{
    if (info.format != V4L2_PIX_FMT_NV12)
        return true;

    VideoBufferInfo out_info;
    out_info.init (info.format, info.width, info.height, info.aligned_width, info.aligned_height);

    if (_swap_flags & (uint32_t)(SwappedBuffer::SwapY)) {
        _swap_offsets[SwappedBuffer::SwapYOffset0] = out_info.offsets[0];
        _swap_offsets[SwappedBuffer::SwapYOffset1] = out_info.size;
        out_info.size += out_info.strides[0] * out_info.aligned_height;
    }

    if (_swap_flags & (uint32_t)(SwappedBuffer::SwapUV)) {
        _swap_offsets[SwappedBuffer::SwapUVOffset0] = out_info.offsets[1];
        _swap_offsets[SwappedBuffer::SwapUVOffset1] = out_info.size;
        out_info.size += out_info.strides[1] * (out_info.aligned_height + 1) / 2;
    }

    if(!init_swap_order (out_info)) {
        XCAM_LOG_ERROR ("DrmBoBufferPool: fix video info faield to init swap order");
        return false;
    }

    info = out_info;
    return true;
}

bool
DrmBoBufferPool::init_swap_order (VideoBufferInfo &info)
{
    if (_swap_flags & (uint32_t)(SwappedBuffer::SwapY)) {
        if ((_swap_init_order & (uint32_t)(SwappedBuffer::OrderYMask)) == (uint32_t)(SwappedBuffer::OrderY0Y1)) {
            info.offsets[0] = _swap_offsets[SwappedBuffer::SwapYOffset0];
        } else if ((_swap_init_order & (uint32_t)(SwappedBuffer::OrderYMask)) ==
                   (uint32_t)(SwappedBuffer::OrderY1Y0)) {
            info.offsets[0] = _swap_offsets[SwappedBuffer::SwapYOffset1];
        } else {
            XCAM_LOG_WARNING ("BufferPool: There's unknown init_swap_order(Y):0x%04x", _swap_init_order);
            return false;
        }
    }

    if (_swap_flags & (uint32_t)(SwappedBuffer::SwapUV)) {
        if ((_swap_init_order & (uint32_t)(SwappedBuffer::OrderUVMask)) == (uint32_t)(SwappedBuffer::OrderUV0UV1)) {
            info.offsets[1] = _swap_offsets[SwappedBuffer::SwapUVOffset0];
        } else if ((_swap_init_order & (uint32_t)(SwappedBuffer::OrderUVMask)) ==
                   (uint32_t)(SwappedBuffer::OrderUV1UV0)) {
            info.offsets[1] = _swap_offsets[SwappedBuffer::SwapUVOffset1];
        } else {
            XCAM_LOG_WARNING ("BufferPool: There's unknown init_swap_order(UV):0x%04x", _swap_init_order);
            return false;
        }
    }

    return true;
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

    SmartPtr<DrmBoBuffer> out_buf = new DrmBoBuffer (info, bo_data);
    XCAM_ASSERT (out_buf.ptr ());
    out_buf->set_swap_info (_swap_flags, _swap_offsets);
    return out_buf;
}

};
