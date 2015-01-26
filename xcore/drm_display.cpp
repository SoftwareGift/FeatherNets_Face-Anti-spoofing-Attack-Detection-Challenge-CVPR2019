/*
 * drm_display.cpp - drm display
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
 * Author: John Ye <john.ye@intel.com>
 */


#include "drm_display.h"
#include "drm_v4l2_buffer.h"

#include <drm.h>
#include <drm_mode.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

#define DEFAULT_DRM_DEVICE "i915"

namespace XCam {

SmartPtr<DrmDisplay> DrmDisplay::_instance(NULL);
Mutex DrmDisplay::_mutex;

SmartPtr<DrmDisplay>
DrmDisplay::instance()
{
    SmartLock lock(_mutex);
    if (_instance.ptr())
        return _instance;
    _instance = new DrmDisplay;
    return _instance;
}

DrmDisplay::DrmDisplay()
    : _module(NULL)
    , _fd (-1)
    , _crtc_index (-1)
    , _crtc_id (0)
    , _con_id (0)
    , _plane_id (0)
    , _format (0)
    , _width (0)
    , _height (0)
    , _capture_buf_type (V4L2_BUF_TYPE_VIDEO_CAPTURE)
{
    xcam_mem_clear(&_compose);

    _fd = drmOpen (DEFAULT_DRM_DEVICE, NULL);
    if (_fd < 0)
        XCAM_LOG_ERROR("failed to open drm device %s", DEFAULT_DRM_DEVICE);
}

XCamReturn
DrmDisplay::get_crtc(drmModeRes *res)
{
    _crtc_index = -1;

    for (int i = 0; i < res->count_crtcs; i++) {
        if (_crtc_id == res->crtcs[i]) {
            _crtc_index = i;
            break;
        }
    }
    XCAM_FAIL_RETURN(ERROR, _crtc_index != -1, XCAM_RETURN_ERROR_PARAM,
                     "CRTC %d not found", _crtc_id);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DrmDisplay::get_connector(drmModeRes *res)
{
    XCAM_FAIL_RETURN(ERROR, res->count_connectors > 0, XCAM_RETURN_ERROR_PARAM,
                     "No connector found");
    _connector = drmModeGetConnector(_fd, _con_id);
    XCAM_FAIL_RETURN(ERROR, _connector, XCAM_RETURN_ERROR_PARAM,
                     "drmModeGetConnector failed: %s\n", strerror(errno));

    return XCAM_RETURN_NO_ERROR;
}


XCamReturn
DrmDisplay::get_plane()
{
    drmModePlaneResPtr planes = drmModeGetPlaneResources(_fd);
    XCAM_FAIL_RETURN(ERROR, planes, XCAM_RETURN_ERROR_PARAM,
                     "failed to query planes: %s", strerror(errno));

    drmModePlanePtr plane = NULL;
    for (uint32_t i = 0; i < planes->count_planes; i++) {
        if (plane) {
            drmModeFreePlane(plane);
            plane = NULL;
        }
        plane = drmModeGetPlane(_fd, planes->planes[i]);
        XCAM_FAIL_RETURN(ERROR, plane, XCAM_RETURN_ERROR_PARAM,
                         "failed to query plane %d: %s", i, strerror(errno));

        if (plane->crtc_id || !(plane->possible_crtcs & (1 << _crtc_index))) {
            continue;
        }

        for (uint32_t j = 0; j < plane->count_formats; j++) {
            // found a plane matching the requested format
            if (plane->formats[j] == _format) {
                _plane_id = plane->plane_id;
                drmModeFreePlane(plane);
                drmModeFreePlaneResources(planes);
                return XCAM_RETURN_NO_ERROR;
            }
        }
    }

    if (plane)
        drmModeFreePlane(plane);

    drmModeFreePlaneResources(planes);

    return XCAM_RETURN_ERROR_PARAM;
}

XCamReturn
DrmDisplay::drm_init(const struct v4l2_pix_format* fmt,
                     const char* module,
                     uint32_t con_id,
                     uint32_t crtc_id,
                     uint32_t width,
                     uint32_t height,
                     uint32_t format,
                     enum v4l2_buf_type capture_buf_type,
                     const struct v4l2_rect* compose)
{
    XCamReturn ret;

    _module = module;
    _con_id = con_id;
    _crtc_id = crtc_id;
    _width = width;
    _height = height;
    _format = format;
    _capture_buf_type = capture_buf_type;
    _compose = *compose;
    _crtc_index = -1;
    _plane_id = 0;
    _connector = NULL;

    drmModeRes *resource = drmModeGetResources(_fd);
    XCAM_FAIL_RETURN(ERROR, resource, XCAM_RETURN_ERROR_PARAM,
                     "failed to query Drm Mode resources: %s", strerror(errno));

    ret = get_crtc(resource);
    XCAM_FAIL_RETURN(ERROR, ret == XCAM_RETURN_NO_ERROR,
                     XCAM_RETURN_ERROR_PARAM,
                     "failed to get CRTC %s", strerror(errno));

    ret = get_connector(resource);
    XCAM_FAIL_RETURN(ERROR, ret == XCAM_RETURN_NO_ERROR,
                     XCAM_RETURN_ERROR_PARAM,
                     "failed to get connector %s", strerror(errno));

    ret = get_plane();
    XCAM_FAIL_RETURN(ERROR, ret == XCAM_RETURN_NO_ERROR,
                     XCAM_RETURN_ERROR_PARAM,
                     "failed to get plane with required format %s", strerror(errno));

    drmModeFreeResources(resource);
    return XCAM_RETURN_NO_ERROR;
}


SmartPtr<V4l2Buffer>
DrmDisplay::create_drm_buf (const struct v4l2_format &format, const uint32_t index, AtomispDevice *device)
{
    struct drm_mode_create_dumb gem;
    struct drm_prime_handle prime;
    struct v4l2_buffer v4l2_buf;
    int ret = 0;

    xcam_mem_clear (&gem);
    xcam_mem_clear (&prime);
    xcam_mem_clear (&v4l2_buf);

    gem.width = format.fmt.pix.bytesperline;
    gem.height = format.fmt.pix.height;
    gem.bpp = 8;
    ret = xcam_device_ioctl (_fd, DRM_IOCTL_MODE_CREATE_DUMB, &gem);
    XCAM_ASSERT (ret >= 0);

    prime.handle = gem.handle;
    ret = xcam_device_ioctl (_fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &prime);
    if (ret < 0) {
        XCAM_LOG_WARNING ("create drm failed on DRM_IOCTL_PRIME_HANDLE_TO_FD");
        return NULL;
    }

    v4l2_buf.index = index;
    v4l2_buf.type = _capture_buf_type;
    v4l2_buf.memory = V4L2_MEMORY_DMABUF;
    v4l2_buf.m.fd = prime.fd;
    v4l2_buf.length = XCAM_MAX (format.fmt.pix.sizeimage, gem.size); // todo check gem.size and format.fmt.pix.length
    XCAM_LOG_DEBUG ("create drm buffer size:%lld", gem.size);
    return new DrmV4l2Buffer (gem.handle, v4l2_buf, format, device, this);
}



XCamReturn
DrmDisplay::drm_setup_framebuffer(SmartPtr<V4l2BufferProxy> &buf,
                                  const struct v4l2_format &format)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    struct drm_prime_handle prime;
    memset(&prime, 0, sizeof (prime));
    prime.fd = buf->get_v4l2_dma_fd();

    ret = (XCamReturn) xcam_device_ioctl(_fd, DRM_IOCTL_PRIME_FD_TO_HANDLE, &prime);
    if (ret) {
        XCAM_LOG_WARNING("FD_TO_PRIME_HANDLE failed: %s\n", strerror(errno));
        return XCAM_RETURN_ERROR_IOCTL;
    }

    uint32_t offsets[4] = { 0 };
    uint32_t pitches[4] = { format.fmt.pix.bytesperline };
    uint32_t bo_handles[4] = { prime.handle };

    uint32_t width = format.fmt.pix.width;
    uint32_t height = format.fmt.pix.height;
    uint32_t fourcc = format.fmt.pix.pixelformat;
    uint32_t fb_handle;

    ret = (XCamReturn) drmModeAddFB2(_fd, width, height, fourcc, bo_handles,
                                     pitches, offsets, &fb_handle, 0);

    _buf_fb_handle[buf->get_v4l2_buf_index()] = fb_handle;

    XCAM_FAIL_RETURN(ERROR, ret == XCAM_RETURN_NO_ERROR, XCAM_RETURN_ERROR_PARAM,
                     "drmModeAddFB2 failed: %s\n", strerror(errno));

    return ret;
}

XCamReturn
DrmDisplay::set_plane(SmartPtr<V4l2BufferProxy> &buf)
{
    XCamReturn ret;
    uint32_t fb_handle = _buf_fb_handle[buf->get_v4l2_buf_index()];

    ret = (XCamReturn) drmModeSetPlane(_fd, _plane_id, _crtc_id,
                                       fb_handle, 0,
                                       _compose.left, _compose.top,
                                       _compose.width, _compose.height,
                                       0, 0, _width << 16, _height << 16);
    XCAM_FAIL_RETURN(ERROR, ret == XCAM_RETURN_NO_ERROR, XCAM_RETURN_ERROR_IOCTL,
                     "failed to set plane via drm: %s\n", strerror(errno));

    drmVBlank vblank;
    vblank.request.type = (drmVBlankSeqType) (DRM_VBLANK_EVENT | DRM_VBLANK_RELATIVE);
    vblank.request.sequence = 1;
    vblank.request.signal = (unsigned long) buf->get_v4l2_buf_index();
    ret = (XCamReturn) drmWaitVBlank(_fd, &vblank);
    XCAM_FAIL_RETURN(ERROR, ret == XCAM_RETURN_NO_ERROR, XCAM_RETURN_ERROR_IOCTL,
                     "failed to wait vblank: %s\n", strerror(errno));

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DrmDisplay::page_flip(SmartPtr<V4l2BufferProxy> &buf)
{
    XCamReturn ret;
    uint32_t fb_handle = _buf_fb_handle[buf->get_v4l2_buf_index()];

    ret = (XCamReturn) drmModePageFlip(_fd, _crtc_id, fb_handle,
                                       DRM_MODE_PAGE_FLIP_EVENT,
                                       (void*)(unsigned long) buf->get_v4l2_buf_index());
    XCAM_FAIL_RETURN(ERROR, ret == XCAM_RETURN_NO_ERROR, XCAM_RETURN_ERROR_IOCTL,
                     "failed on page flip: %s\n", strerror(errno));

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DrmDisplay::display_buffer(SmartPtr<V4l2BufferProxy> &buf)
{
    return _plane_id ? set_plane(buf) : page_flip(buf);
}

};


