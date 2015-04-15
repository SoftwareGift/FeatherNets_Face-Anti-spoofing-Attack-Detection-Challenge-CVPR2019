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
#include "isp_controller.h"

namespace XCam {

class X3aStats;

class PollCallback {
public:
    virtual ~PollCallback() {}
    virtual XCamReturn poll_buffer_ready (SmartPtr<V4l2BufferProxy> &buf) = 0;
    virtual XCamReturn poll_buffer_failed (int64_t timestamp, const char *msg) = 0;
    virtual XCamReturn poll_3a_stats_ready (SmartPtr<X3aStats> &stats) = 0;
    virtual XCamReturn poll_dvs_stats_ready() = 0;
};

class V4l2Device;
class V4l2SubDevice;
class X3aStatisticsQueue;
class EventPollThread;
class CapturePollThread;

class PollThread
{
    friend class EventPollThread;
    friend class CapturePollThread;
public:
    explicit PollThread ();
    ~PollThread ();

    bool set_capture_device (SmartPtr<V4l2Device> &dev);
    bool set_event_device (SmartPtr<V4l2SubDevice> &sub_dev);
    bool set_isp_controller (SmartPtr<IspController>  &isp);
    bool set_callback (PollCallback *callback);

    XCamReturn start();
    XCamReturn stop ();

protected:
    XCamReturn poll_subdev_event_loop ();
    XCamReturn poll_buffer_loop ();

    XCamReturn handle_events (struct v4l2_event &event);
    XCamReturn handle_3a_stats_event (struct v4l2_event &event);

private:
    XCamReturn init_3a_stats_pool ();
    XCamReturn capture_3a_stats (SmartPtr<X3aStats> &stats);


private:
    XCAM_DEAD_COPY (PollThread);

private:
    static const int default_subdev_event_timeout;
    static const int default_capture_event_timeout;

    SmartPtr<EventPollThread>        _event_loop;
    SmartPtr<CapturePollThread>      _capture_loop;

    SmartPtr<V4l2SubDevice>          _event_dev;
    SmartPtr<X3aStatsPool>           _3a_stats_pool;

    SmartPtr<V4l2Device>             _capture_dev;
    SmartPtr<IspController>          _isp_controller;

    PollCallback                    *_callback;
};

};

#endif //XCAM_POLL_THREAD_H
