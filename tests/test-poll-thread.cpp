/*
 * main.cpp - test
 *
 *  Copyright (c) 2014 Intel Corporation
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

#include "device_manager.h"
#include "atomisp_device.h"
#include "x3a_analyzer_aiq.h"
#include "isp_controller.h"
#include "isp_image_processor.h"
#include <unistd.h>
#include <signal.h>
#include "test_common.h"

#if HAVE_LIBDRM
#include "drm_display.h"
#endif

using namespace XCam;

class PollCB: public PollCallback {
public:

#if HAVE_LIBDRM
    PollCB(SmartPtr<DrmDisplay> &drm_dev, struct v4l2_format &format)
        : _file (NULL)
        , _format(format)
        , _drm_dev(drm_dev)

    {
        open_file();
    };
#else
    PollCB(struct v4l2_format &format)
        : _file (NULL),
          _format(format)
    {
        open_file();
    };
#endif

    ~PollCB() {
        close_file ();
    };
    XCamReturn poll_buffer_ready (SmartPtr<VideoBuffer> &buf);
    XCamReturn poll_buffer_failed (int64_t timestamp, const char *msg)
    {
        XCAM_UNUSED(timestamp);
        XCAM_UNUSED(msg);
        XCAM_LOG_DEBUG("%s", __FUNCTION__);
        return XCAM_RETURN_NO_ERROR;
    }
    XCamReturn x3a_stats_ready (const SmartPtr<X3aStats> &stats) {
        XCAM_UNUSED(stats);
        XCAM_LOG_DEBUG("%s", __FUNCTION__);
        return XCAM_RETURN_NO_ERROR;
    }
    XCamReturn dvs_stats_ready() {
        XCAM_LOG_DEBUG("%s", __FUNCTION__);
        return XCAM_RETURN_NO_ERROR;
    }

private:
    void open_file ();
    void close_file ();
    size_t dump_to_file(const void *buf, size_t nbyte);

    FILE      *_file;
    struct v4l2_format _format;
#if HAVE_LIBDRM
    SmartPtr<DrmDisplay> _drm_dev;
#endif
};

XCamReturn
PollCB::poll_buffer_ready (SmartPtr<VideoBuffer> &buf) {

    SmartPtr<VideoBuffer> base = buf;
    XCAM_LOG_DEBUG("%s", __FUNCTION__);
    FPS_CALCULATION (fps_buf, 30);

    // dump_to_file( (void*) buf->get_v4l2_userptr(),
    //               buf->get_v4l2_buf_length()
    //             );

#if HAVE_LIBDRM
    //if (!_drm_dev->has_frame_buffer (base))
    _drm_dev->render_setup_frame_buffer (base);

    _drm_dev->render_buffer (base);
#endif

    return XCAM_RETURN_NO_ERROR;
}


void
PollCB::open_file ()
{
    if (_file)
        return;
    _file = fopen ("capture_buffer.nv12", "wb");
}

void
PollCB::close_file ()
{
    if (_file)
        fclose (_file);
    _file = NULL;
}

size_t
PollCB::dump_to_file (const void *buf, size_t nbyte)
{
    if (!_file)
        return 0;
    return fwrite(buf, nbyte, 1, _file);
}


#define V4L2_CAPTURE_MODE_STILL   0x2000
#define V4L2_CAPTURE_MODE_VIDEO   0x4000
#define V4L2_CAPTURE_MODE_PREVIEW 0x8000

static Mutex g_mutex;
static Cond  g_cond;
static bool  g_stop = false;

void dev_stop_handler(int sig)
{
    XCAM_UNUSED (sig);

    SmartLock locker (g_mutex);
    g_stop = true;
    g_cond.broadcast ();

    // exit(0);
}

int main (int argc, const char *argv[])
{
    (void)argv;
    (void)argc; // suppress unused variable warning

    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<V4l2Device> device = new AtomispDevice (DEFAULT_CAPTURE_DEVICE);
    SmartPtr<V4l2SubDevice> event_device = new V4l2SubDevice (DEFAULT_EVENT_DEVICE);
    SmartPtr<IspController> isp_controller = new IspController (device);
    SmartPtr<ImageProcessor> processor = new IspImageProcessor (isp_controller);

    device->set_sensor_id (0);
    device->set_capture_mode (V4L2_CAPTURE_MODE_VIDEO);
    //device->set_mem_type (V4L2_MEMORY_MMAP);
    device->set_mem_type (V4L2_MEMORY_DMABUF);
    device->set_buffer_count (8);
    device->set_framerate (25, 1);
    ret = device->open ();
    CHECK (ret, "device(%s) open failed", device->get_device_name());
    //ret = device->set_format (1920, 1080, V4L2_PIX_FMT_NV12, V4L2_FIELD_NONE, 1920 * 2);
    ret = device->set_format (1920, 1080, V4L2_PIX_FMT_YUYV, V4L2_FIELD_NONE, 1920 * 2);
    CHECK (ret, "device(%s) set format failed", device->get_device_name());


    ret = event_device->open ();
    CHECK (ret, "event device(%s) open failed", event_device->get_device_name());
    int event = V4L2_EVENT_ATOMISP_3A_STATS_READY;
    ret = event_device->subscribe_event (event);
    CHECK_CONTINUE (
        ret,
        "device(%s) subscribe event(%d) failed",
        event_device->get_device_name(), event);
    event = V4L2_EVENT_FRAME_SYNC;
    ret = event_device->subscribe_event (event);
    CHECK_CONTINUE (
        ret,
        "device(%s) subscribe event(%d) failed",
        event_device->get_device_name(), event);
    ret = event_device->start();
    CHECK (ret, "event device start failed");

    struct v4l2_format format;
    device->get_format(format);

#if HAVE_LIBDRM
    AtomispDevice* atom_isp_dev = (AtomispDevice*)device.ptr();
    SmartPtr<DrmDisplay> drmdisp = DrmDisplay::instance();
    struct v4l2_rect rect = { 0, 0, (int)format.fmt.pix.width, (int)format.fmt.pix.height };
    drmdisp->render_init(
        0,
        0,
        1920,
        1080,
        format.fmt.pix.pixelformat,
        &rect);
    atom_isp_dev->set_drm_display(drmdisp);

    ret = device->start();
    CHECK (ret, "capture device start failed");
    SmartPtr<PollThread> poll_thread = new PollThread();
    PollCB* poll_cb = new PollCB(drmdisp, format);
#else
    ret = device->start();
    CHECK(ret, "capture device start failed");
    SmartPtr<PollThread> poll_thread = new PollThread();
    PollCB* poll_cb = new PollCB(format);
#endif

    poll_thread->set_capture_device(device);
    poll_thread->set_event_device(event_device);
    poll_thread->set_isp_controller(isp_controller);
    poll_thread->set_poll_callback(poll_cb);

    signal(SIGINT, dev_stop_handler);

    poll_thread->start();
    CHECK (ret, "poll thread start failed");

    // wait for interruption
    {
        SmartLock locker (g_mutex);
        while (!g_stop)
            g_cond.wait (g_mutex);
    }

    ret = poll_thread->stop();
    CHECK_CONTINUE (ret, "poll thread stop failed");
    device->close ();
    event_device->close ();

    return 0;
}
