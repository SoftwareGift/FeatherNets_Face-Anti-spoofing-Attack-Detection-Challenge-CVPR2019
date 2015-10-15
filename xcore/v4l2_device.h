/*
 * v4l2_device.h - v4l2 device
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

#ifndef XCAM_V4L2_DEVICE_H
#define XCAM_V4L2_DEVICE_H

#include "xcam_utils.h"
#include "smartptr.h"
#include <linux/videodev2.h>
#include <list>
#include <vector>

extern "C" {
    struct v4l2_event;
    struct v4l2_format;
    struct v4l2_fmtdesc;
    struct v4l2_frmsizeenum;
}

namespace XCam {

class V4l2Buffer;

class V4l2Device {
    friend class V4l2BufferProxy;
    typedef std::vector<SmartPtr<V4l2Buffer>> BufferPool;

public:
    V4l2Device (const char *name = NULL);
    virtual ~V4l2Device ();

    // before device open
    bool set_device_name (const char *name);
    bool set_sensor_id (int id);
    bool set_capture_mode (uint32_t capture_mode);

    int get_fd () const {
        return _fd;
    }
    const char *get_device_name () const {
        return _name;
    }
    bool is_opened () const    {
        return (_fd != -1);
    }
    bool is_activated () const {
        return _active;
    }

    // set_mem_type must before set_format
    bool set_mem_type (enum v4l2_memory type);
    enum v4l2_memory get_mem_type () const {
        return _memory_type;
    }
    enum v4l2_buf_type get_capture_buf_type () const {
        return _capture_buf_type;
    }
    void get_size (uint32_t &width, uint32_t &height) const {
        width = _format.fmt.pix.width;
        height = _format.fmt.pix.height;
    }
    uint32_t get_pixel_format () const {
        return _format.fmt.pix.pixelformat;
    }

    bool set_buffer_count (uint32_t buf_count);

    // set_framerate must before set_format
    bool set_framerate (uint32_t n, uint32_t d);
    void get_framerate (uint32_t &n, uint32_t &d);

    XCamReturn open ();
    XCamReturn close ();
    // set_format
    XCamReturn get_format (struct v4l2_format &format);
    XCamReturn set_format (struct v4l2_format &format);
    XCamReturn set_format (
        uint32_t width, uint32_t height, uint32_t pixelformat,
        enum v4l2_field field = V4L2_FIELD_NONE, uint32_t bytes_perline = 0);

    std::list<struct v4l2_fmtdesc> enum_formats ();

    virtual XCamReturn start ();
    virtual XCamReturn stop ();

    int poll_event (int timeout_msec);
    XCamReturn dequeue_buffer (SmartPtr<V4l2Buffer> &buf);
    XCamReturn queue_buffer (SmartPtr<V4l2Buffer> &buf);

    // use as less as possible
    virtual int io_control (int cmd, void *arg);

protected:

    //virtual functions, handle private actions on set_format
    virtual XCamReturn pre_set_format (struct v4l2_format &format);
    virtual XCamReturn post_set_format (struct v4l2_format &format);
    virtual XCamReturn allocate_buffer (
        SmartPtr<V4l2Buffer> &buf,
        const struct v4l2_format &format,
        const uint32_t index);

private:
    XCamReturn request_buffer ();
    XCamReturn init_buffer_pool ();
    XCamReturn fini_buffer_pool ();

    XCAM_DEAD_COPY (V4l2Device);

protected:
    char               *_name;
    int                 _fd;
    int32_t             _sensor_id;
    uint32_t            _capture_mode;
    enum v4l2_buf_type  _capture_buf_type;
    enum v4l2_memory    _memory_type;

    struct v4l2_format  _format;
    uint32_t            _fps_n;
    uint32_t            _fps_d;

    bool                _active;

    // buffer pool
    BufferPool          _buf_pool;
    uint32_t            _buf_count;

    XCamReturn buffer_new();
    XCamReturn buffer_del();
};

class V4l2SubDevice
    : public V4l2Device
{
public:
    explicit V4l2SubDevice (const char *name = NULL);

    XCamReturn subscribe_event (int event);
    XCamReturn unsubscribe_event (int event);
    XCamReturn dequeue_event (struct v4l2_event &event);

    virtual XCamReturn start ();
    virtual XCamReturn stop ();

private:
    XCAM_DEAD_COPY (V4l2SubDevice);
};

};
#endif // XCAM_V4L2_DEVICE_H

