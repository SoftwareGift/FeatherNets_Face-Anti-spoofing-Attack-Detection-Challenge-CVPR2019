/*
 * atomisp_device.h - atomisp device
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
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#ifndef XCAM_ATOMISP_DEVICE_H
#define XCAM_ATOMISP_DEVICE_H

#include "xcam_common.h"
#include "xcam_defs.h"
#include "v4l2_device.h"

namespace XCam {

class AtomispDevice
    : public V4l2Device
{
    friend class DrmV4l2Buffer;

public:
    explicit AtomispDevice (const char *name = NULL);
    ~AtomispDevice ();

protected:
    virtual XCamReturn pre_set_format (struct v4l2_format &format);
    virtual XCamReturn allocate_buffer (
        SmartPtr<V4l2Buffer> &buf,
        const struct v4l2_format &format,
        const uint32_t index);

private:
    XCAM_DEAD_COPY (AtomispDevice);

#if HAVE_LIBDRM
    int get_drm_handle () const {
        return _drm_handle;
    }
    SmartPtr<V4l2Buffer> create_drm_buf (const struct v4l2_format &format, const uint32_t index);
#endif

private:
    int _drm_handle;
};

};

#endif //XCAM_ATOMISP_DEVICE_H
