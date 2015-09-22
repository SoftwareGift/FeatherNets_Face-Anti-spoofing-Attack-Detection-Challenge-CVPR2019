/*
 * scaled_buffer_pool.cpp -  video scaled buffer pool
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
 * Author: Zong Wei <wei.zong@intel.com>
 */

#include "scaled_buffer_pool.h"

namespace XCam {

ScaledVideoBuffer::ScaledVideoBuffer (const VideoBufferInfo &info, const XCamVideoBufferInfo &scaled_info, const SmartPtr<DrmBoData> &data)
    : DrmBoBuffer (info, data)
    , _video_info (scaled_info)
{
}


XCamReturn
ScaledVideoBuffer::get_scaled_buffer (XCamVideoBuffer& buffer)
{
    SmartPtr<BufferData> data = get_buffer_data ();
    XCAM_ASSERT(data.ptr());

    buffer.data = data->map ();
    buffer.info = _video_info;
    buffer.timestamp = get_timestamp ();

    return XCAM_RETURN_NO_ERROR;
}

ScaledVideoBufferPool::ScaledVideoBufferPool (SmartPtr<DrmDisplay> &display)
    : DrmBoBufferPool (display)
{
}

bool
ScaledVideoBufferPool::fixate_video_info (VideoBufferInfo &info)
{
    _video_info.format = info.format;
    _video_info.color_bits = info.color_bits;
    _video_info.width = info.width;
    _video_info.height = info.height;
    _video_info.aligned_width = info.aligned_width;
    _video_info.aligned_height = info.aligned_height;
    _video_info.size = info.size;
    _video_info.components = info.components;

    for (uint8_t i = 0; i < XCAM_VIDEO_MAX_COMPONENTS; i++) {
        _video_info.strides[i] = info.strides[i];
        _video_info.offsets[i] = info.offsets[i];
    }
    return true;
}

SmartPtr<BufferProxy>
ScaledVideoBufferPool::create_buffer_from_data (SmartPtr<BufferData> &data)
{
    SmartPtr<DrmBoData> buffer_data = data.dynamic_cast_ptr<DrmBoData> ();
    const VideoBufferInfo & info = get_video_info ();
    const XCamVideoBufferInfo & scaled_info = get_scaled_video_info ();
    XCAM_ASSERT (data.ptr ());

    return new ScaledVideoBuffer (info, scaled_info, buffer_data);
}

};
