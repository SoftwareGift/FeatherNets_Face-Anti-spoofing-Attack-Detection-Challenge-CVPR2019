/*
 * drm_display.h - drm display
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

#ifndef XCAM_DRM_DISPLAY_H
#define XCAM_DRM_DISPLAY_H

#include "xcam_common.h"
#include "smartptr.h"
#include "xcam_mutex.h"
#include "atomisp_device.h"
#include "v4l2_buffer_proxy.h"

#include <errno.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

#include <linux/videodev2.h>
#include <list>
#include <vector>
#include <map>

extern "C" {
    struct v4l2_pix_format;
}

namespace XCam {

class AtomispDevice;

class DrmDisplay {
public:
    virtual ~DrmDisplay() {
        if (_fd > 0)
            drmClose(_fd);
    };

    static SmartPtr<DrmDisplay> instance();

    XCamReturn drm_init(const struct v4l2_pix_format *fmt,
                        const char* module,
                        uint32_t con_id,
                        uint32_t crtc_id,
                        uint32_t width,
                        uint32_t height,
                        uint32_t format,
                        enum v4l2_buf_type capture_buf_type,
                        struct v4l2_rect compose);

    bool has_fb_handle(SmartPtr<V4l2BufferProxy> &buf) {
        return _buf_fb_handle.find(buf->get_v4l2_buf_index()) != _buf_fb_handle.end();
    };
    XCamReturn drm_setup_framebuffer(SmartPtr<V4l2BufferProxy> &buf, const struct v4l2_format &format);
    XCamReturn display_buffer(SmartPtr<V4l2BufferProxy> &buf);

    int get_drm_handle() const {
        return _fd;
    };

    SmartPtr<V4l2Buffer> create_drm_buf (const struct v4l2_format &format, const uint32_t index, AtomispDevice *device);

private:
    DrmDisplay();

    XCamReturn get_crtc(drmModeRes *res);
    XCamReturn get_connector(drmModeRes *res);
    XCamReturn get_plane();
    XCamReturn set_plane(SmartPtr<V4l2BufferProxy> &buf);
    XCamReturn page_flip(SmartPtr<V4l2BufferProxy> &buf);

private:
    const char *_module;
    int _fd;

    int _crtc_index;
    unsigned int _crtc_id;
    unsigned int _con_id;
    unsigned int _plane_id;
    drmModeConnector *_connector;

    drmModeModeInfo _mode;

    unsigned int _format;
    unsigned int _width;
    unsigned int _height;

    enum v4l2_buf_type _capture_buf_type;

    struct v4l2_rect _compose;

    std::map<uint32_t, uint32_t> _buf_fb_handle;

private:
    XCAM_DEAD_COPY (DrmDisplay);

private:
    static SmartPtr<DrmDisplay> _instance;
    static Mutex                _mutex;
};

};
#endif // XCAM_DRM_DISPLAY_H

