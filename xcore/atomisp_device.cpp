/*
 * atomisp_device.cpp - atomisp device
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

#include "atomisp_device.h"
#include "v4l2_buffer_proxy.h"
#include <linux/v4l2-subdev.h>

namespace XCam {

AtomispDevice::AtomispDevice (const char *name)
    : V4l2Device (name)
{
}

AtomispDevice::~AtomispDevice ()
{
}

XCamReturn
AtomispDevice::pre_set_format (struct v4l2_format &format)
{
    uint32_t fps_n = 0, fps_d = 0;
    struct v4l2_subdev_format subdev_fmt;

    // set framerate by subdev
    this->get_framerate (fps_n, fps_d);
    if (fps_n != 0 && fps_d != 0) {
        struct v4l2_subdev_frame_interval frame_intvl;
        xcam_mem_clear (frame_intvl);
        if (io_control (VIDIOC_SUBDEV_G_FRAME_INTERVAL, &frame_intvl) < 0) {
            XCAM_LOG_WARNING ("atomisp device(%s) get framerate failed ", XCAM_STR (get_device_name()));
        } else {
            frame_intvl.interval.denominator = fps_n;
            frame_intvl.interval.numerator = fps_d;
            if (io_control (VIDIOC_SUBDEV_S_FRAME_INTERVAL, &frame_intvl) < 0) {
                XCAM_LOG_WARNING ("atomisp device(%s) set framerate failed", XCAM_STR (get_device_name()));
            }
        }
    }

    // negotiate and set sensor output format by subdev
    xcam_mem_clear (subdev_fmt);
    subdev_fmt.pad = 0;
    subdev_fmt.which = V4L2_SUBDEV_FORMAT_TRY;
    subdev_fmt.format.width = format.fmt.pix.width;
    subdev_fmt.format.height = format.fmt.pix.height;
    subdev_fmt.format.field = V4L2_FIELD_NONE;
    if (format.fmt.pix.pixelformat == V4L2_PIX_FMT_SGRBG12) {
        subdev_fmt.format.code = V4L2_MBUS_FMT_SRGGB12_1X12;
    } else {
        subdev_fmt.format.code = V4L2_MBUS_FMT_SRGGB10_1X10;
    }

    if (io_control(VIDIOC_SUBDEV_S_FMT, &subdev_fmt) < 0) {
        XCAM_LOG_ERROR ("atomisp device(%s) try subdev format failed", XCAM_STR (get_device_name()));
        return XCAM_RETURN_ERROR_IOCTL;
    }
    XCAM_LOG_INFO ("target subdev format (%dx%d, code %d)",
                   subdev_fmt.format.width,
                   subdev_fmt.format.height,
                   subdev_fmt.format.code);

    subdev_fmt.which = V4L2_SUBDEV_FORMAT_ACTIVE;
    if (io_control (VIDIOC_SUBDEV_G_FMT, &subdev_fmt) < 0) {
        XCAM_LOG_ERROR ("atomisp device(%s) get subdev format failed", XCAM_STR (get_device_name()));
    }
    XCAM_LOG_INFO ("negotiated subdev format (%dx%d, code %d)",
                   subdev_fmt.format.width,
                   subdev_fmt.format.height,
                   subdev_fmt.format.code);

    if (io_control(VIDIOC_SUBDEV_S_FMT, &subdev_fmt) < 0) {
        XCAM_LOG_ERROR ("atomisp device(%s) set subdev format failed", XCAM_STR (get_device_name()));
        return XCAM_RETURN_ERROR_IOCTL;
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
AtomispDevice::allocate_buffer (
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
            XCAM_LOG_WARNING ("atomisp device(%s) allocate buffer failed", XCAM_STR (get_device_name()));
            return XCAM_RETURN_ERROR_MEM;
        }
        return XCAM_RETURN_NO_ERROR;
    }
#endif

    return V4l2Device::allocate_buffer (buf, format, index);
}

};
