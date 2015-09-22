/*
 * scaled_buffer_pool.h -  video scaled buffer pool
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

#ifndef XCAM_SCALED_BUFFER_POOL_H
#define XCAM_SCALED_BUFFER_POOL_H

#include "xcam_utils.h"
#include "drm_bo_buffer.h"
#include <base/xcam_3a_stats.h>

namespace XCam {

class ScaledVideoBuffer
    : public DrmBoBuffer
{
    friend class ScaledVideoBufferPool;
public:
    XCamReturn get_scaled_buffer (XCamVideoBuffer& buffer);

protected:
    explicit ScaledVideoBuffer (const VideoBufferInfo &info, const XCamVideoBufferInfo &scaled_info, const SmartPtr<DrmBoData> &data);
    XCAM_DEAD_COPY (ScaledVideoBuffer);

private:
    XCamVideoBufferInfo _video_info;
};

class ScaledVideoBufferPool
    : public DrmBoBufferPool
{
public:
    explicit ScaledVideoBufferPool (SmartPtr<DrmDisplay> &display);
    XCamVideoBufferInfo &get_scaled_video_info () {
        return _video_info;
    }

protected:
    virtual bool fixate_video_info (VideoBufferInfo &info);
    virtual SmartPtr<BufferProxy> create_buffer_from_data (SmartPtr<BufferData> &data);

private:
    XCAM_DEAD_COPY (ScaledVideoBufferPool);

private:
    XCamVideoBufferInfo _video_info;
};

};

#endif //XCAM_SCALED_BUFFER_POOL_H

