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
#if HAVE_LIBDRM
#include <drm.h>
#include <drm_mode.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

#define DEFAULT_DRM_DEVICE "i915"
#endif

namespace XCam {

AtomispDevice::AtomispDevice (const char *name)
    : V4l2Device (name)
    , _drm_handle (-1)
{
}

AtomispDevice::~AtomispDevice ()
{
#if HAVE_LIBDRM
    if (_drm_handle > 0)
        drmClose (_drm_handle);
#endif
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

        xcam_mem_clear (&frame_intvl);
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

    xcam_mem_clear (&subdev_fmt);
    subdev_fmt.pad = 0;
    subdev_fmt.which = V4L2_SUBDEV_FORMAT_ACTIVE;
    subdev_fmt.format.width = format.fmt.pix.width + 32;
    subdev_fmt.format.height = format.fmt.pix.height + 17;
    subdev_fmt.format.code = V4L2_MBUS_FMT_SRGGB10_1X10; //depends on sensor V4L2_MBUS_FMT_UYVY8_1X16;
    subdev_fmt.format.field = V4L2_FIELD_NONE;

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
    if (get_mem_type () == V4L2_MEMORY_DMABUF) {
        buf = create_drm_buf (format, index);
        if (!buf.ptr()) {
            XCAM_LOG_WARNING ("atomisp device(%s) allocate buffer failed", XCAM_STR (get_device_name()));
            return XCAM_RETURN_ERROR_MEM;
        }
        return XCAM_RETURN_NO_ERROR;
    }
#endif

    return V4l2Device::allocate_buffer (buf, format, index);
}

#if HAVE_LIBDRM

class DrmV4l2Buffer
    : public V4l2Buffer
{
public:
    explicit DrmV4l2Buffer (
        uint32_t gem_handle,
        const struct v4l2_buffer &buf,
        const struct v4l2_format &format,
        AtomispDevice *device
    )
        : V4l2Buffer (buf, format)
        , _gem_handle (gem_handle)
        , _device (device)
    {}
    ~DrmV4l2Buffer ();

private:
    uint32_t       _gem_handle;
    AtomispDevice *_device;
};

DrmV4l2Buffer::~DrmV4l2Buffer ()
{
    XCAM_ASSERT (_device);
    int handle = _device->get_drm_handle ();
    if (handle > 0) {
        struct drm_mode_destroy_dumb gem;
        xcam_mem_clear (&gem);
        gem.handle = _gem_handle;
        xcam_device_ioctl (handle, DRM_IOCTL_MODE_DESTROY_DUMB, &gem);
    }
}


SmartPtr<V4l2Buffer>
AtomispDevice::create_drm_buf (const struct v4l2_format &format, const uint32_t index)
{
    struct drm_mode_create_dumb gem;
    struct drm_prime_handle prime;
    struct v4l2_buffer v4l2_buf;
    int ret = 0;

    xcam_mem_clear (&gem);
    xcam_mem_clear (&prime);
    xcam_mem_clear (&v4l2_buf);

    if (_drm_handle < 0)
        _drm_handle = drmOpen (DEFAULT_DRM_DEVICE, NULL);
    if (_drm_handle < 0) {
        XCAM_LOG_WARNING ("open drm device(%s) failed", DEFAULT_DRM_DEVICE);
        return NULL;
    }

    gem.width = format.fmt.pix.bytesperline;
    gem.height = format.fmt.pix.height;
    gem.bpp = 8;
    ret = xcam_device_ioctl (_drm_handle, DRM_IOCTL_MODE_CREATE_DUMB, &gem);
    XCAM_ASSERT (ret >= 0);

    prime.handle = gem.handle;
    ret = xcam_device_ioctl (_drm_handle, DRM_IOCTL_PRIME_HANDLE_TO_FD, &prime);
    if (ret < 0) {
        XCAM_LOG_WARNING ("create drm failed on DRM_IOCTL_PRIME_HANDLE_TO_FD");
        return NULL;
    }

    v4l2_buf.index = index;
    v4l2_buf.type = get_capture_buf_type ();
    v4l2_buf.memory = V4L2_MEMORY_DMABUF;
    v4l2_buf.m.fd = prime.fd;
    v4l2_buf.length = XCAM_MAX (format.fmt.pix.sizeimage, gem.size); // todo check gem.size and format.fmt.pix.length
    XCAM_LOG_DEBUG ("create drm buffer size:%lld", gem.size);
    return new DrmV4l2Buffer (gem.handle, v4l2_buf, format, this);
}

#endif

};
