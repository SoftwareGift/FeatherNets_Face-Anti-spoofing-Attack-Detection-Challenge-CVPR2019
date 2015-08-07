/*
 * uvc_device.cpp - uvc device
 *
 *  Copyright (c) 2014-2015 Intel Corporation
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
 * Author: Sameer Kibey <sameer.kibey@intel.com>
 */

#include "uvc_device.h"
#include "v4l2_buffer_proxy.h"
#include <linux/v4l2-subdev.h>

namespace XCam {

UVCDevice::UVCDevice (const char *name)
    : V4l2Device (name)
{
}

UVCDevice::~UVCDevice ()
{
}

XCamReturn
UVCDevice::allocate_buffer (
    SmartPtr<V4l2Buffer> &buf,
    const struct v4l2_format &format,
    const uint32_t index)
{
#if HAVE_LIBDRM
    if (!_drm_disp.ptr()) {
        _drm_disp = DrmDisplay::instance ();
    }

    if (get_mem_type () == V4L2_MEMORY_DMABUF && _drm_disp.ptr () != NULL) {
        buf = _drm_disp->create_drm_buf (format, index, get_capture_buf_type ());
        if (!buf.ptr()) {
            XCAM_LOG_WARNING ("uvc device(%s) allocate buffer failed", XCAM_STR (get_device_name()));
            return XCAM_RETURN_ERROR_MEM;
        }
        return XCAM_RETURN_NO_ERROR;
    }
#endif

    return V4l2Device::allocate_buffer (buf, format, index);
}

};
