/*
 * uvc_device.h - uvc device
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

#ifndef XCAM_UVC_DEVICE_H
#define XCAM_UVC_DEVICE_H

#include <xcam_std.h>
#include <v4l2_device.h>
#if HAVE_LIBDRM
#include <drm_display.h>
#endif

namespace XCam {

#if HAVE_LIBDRM
class DrmDisplay;
#endif

class UVCDevice
    : public V4l2Device
{
    friend class DrmV4l2Buffer;

public:
    explicit UVCDevice (const char *name = NULL);
    ~UVCDevice ();

#if HAVE_LIBDRM
    void set_drm_display(SmartPtr<DrmDisplay> &drm_disp) {
        _drm_disp = drm_disp;
    };
#endif

protected:
    virtual XCamReturn allocate_buffer (
        SmartPtr<V4l2Buffer> &buf,
        const struct v4l2_format &format,
        const uint32_t index);

private:
    XCAM_DEAD_COPY (UVCDevice);

#if HAVE_LIBDRM
private:
    SmartPtr<DrmDisplay> _drm_disp;
#endif
};

};

#endif //XCAM_UVC_DEVICE_H
