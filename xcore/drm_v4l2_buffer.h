/*
 * drm_v4l2_buffer.h - drm v4l2 buffer
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
 * Author: John Ye <john.ye@intel.com>
 */

#ifndef XCAM_DRM_V4L2_BUFFER_H
#define XCAM_DRM_V4L2_BUFFER_H

#include <xcam_std.h>
#include <v4l2_buffer_proxy.h>
#include <drm_display.h>

namespace XCam {

class AtomispDevice;

class DrmV4l2Buffer
    : public V4l2Buffer
{
public:
    explicit DrmV4l2Buffer (
        uint32_t gem_handle,
        const struct v4l2_buffer &buf,
        const struct v4l2_format &format,
        SmartPtr<DrmDisplay> &display
    )
        : V4l2Buffer (buf, format)
        , _gem_handle (gem_handle)
        , _display (display)
    {}
    ~DrmV4l2Buffer ();

private:
    XCAM_DEAD_COPY (DrmV4l2Buffer);

private:
    uint32_t       _gem_handle;
    SmartPtr<DrmDisplay> _display;
};

};

#endif // XCAM_DRM_V4L2_BUFFER_H
