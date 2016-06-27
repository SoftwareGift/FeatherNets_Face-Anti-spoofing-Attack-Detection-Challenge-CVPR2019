/*
 * poll_thread.h - poll thread for event and buffer
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

#ifndef XCAM_POLL_THREAD_H
#define XCAM_POLL_THREAD_H

#include "xcam_utils.h"
#include "xcam_mutex.h"
#include "x3a_event.h"
#include "v4l2_buffer_proxy.h"
#include "x3a_stats_pool.h"
#include "v4l2_device.h"
#include "stats_callback_interface.h"

namespace XCam {

class PollCallback
{
public:
    PollCallback () {}
    virtual ~PollCallback() {}
    virtual XCamReturn poll_buffer_ready (SmartPtr<VideoBuffer> &buf) = 0;
    virtual XCamReturn poll_buffer_failed (int64_t timestamp, const char *msg) = 0;

private:
    XCAM_DEAD_COPY (PollCallback);

};

class V4l2Device;
class V4l2SubDevice;
class EventPollThread;
class CapturePollThread;

class PollThread
{
    friend class EventPollThread;
    friend class CapturePollThread;
    friend class FakePollThread;
public:
    explicit PollThread ();
    virtual ~PollThread ();

    bool set_capture_device (SmartPtr<V4l2Device> &dev);
    bool set_event_device (SmartPtr<V4l2SubDevice> &sub_dev);
    bool set_poll_callback (PollCallback *callback);
    bool set_stats_callback (StatsCallback *callback);

    virtual XCamReturn start();
    virtual XCamReturn stop ();

protected:
    XCamReturn poll_subdev_event_loop ();
    virtual XCamReturn poll_buffer_loop ();

    virtual XCamReturn handle_events (struct v4l2_event &event);
    XCamReturn handle_3a_stats_event (struct v4l2_event &event);

private:
    virtual XCamReturn init_3a_stats_pool ();
    virtual XCamReturn capture_3a_stats (SmartPtr<X3aStats> &stats);

private:
    XCAM_DEAD_COPY (PollThread);

private:
    static const int default_subdev_event_timeout;
    static const int default_capture_event_timeout;

    SmartPtr<EventPollThread>        _event_loop;
    SmartPtr<CapturePollThread>      _capture_loop;

    SmartPtr<V4l2SubDevice>          _event_dev;
    SmartPtr<V4l2Device>             _capture_dev;

    PollCallback                    *_poll_callback;
    StatsCallback                   *_stats_callback;
};

};

#endif //XCAM_POLL_THREAD_H
