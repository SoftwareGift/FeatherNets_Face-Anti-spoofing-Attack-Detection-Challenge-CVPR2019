/*
 * drm_v4l2_buffer.cpp - drm buffer
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

#include "drm_v4l2_buffer.h"

namespace XCam {

DrmV4l2Buffer::~DrmV4l2Buffer ()
{
    XCAM_ASSERT (_display.ptr());
    int handle = _display->get_drm_handle ();
    if (handle > 0) {
        struct drm_mode_destroy_dumb gem;
        xcam_mem_clear (gem);
        gem.handle = _gem_handle;
        xcam_device_ioctl (handle, DRM_IOCTL_MODE_DESTROY_DUMB, &gem);
    }
}

};
