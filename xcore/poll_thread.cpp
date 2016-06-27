/*
 * poll_thread.cpp - poll thread for event and buffer
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

#include "poll_thread.h"
#include "xcam_thread.h"
#include <unistd.h>

namespace XCam {

class PollThread;

class EventPollThread
    : public Thread
{
public:
    EventPollThread (PollThread *poll)
        : Thread ("event_poll")
        , _poll (poll)
    {}

protected:
    virtual bool started () {
        XCamReturn ret = _poll->init_3a_stats_pool ();
        if (ret != XCAM_RETURN_NO_ERROR)
            return false;
        return true;
    }
    virtual bool loop () {
        XCamReturn ret = _poll->poll_subdev_event_loop ();

        if (ret == XCAM_RETURN_NO_ERROR || ret == XCAM_RETURN_ERROR_TIMEOUT)
            return true;
        return false;
    }

private:
    PollThread   *_poll;
};

class CapturePollThread
    : public Thread
{
public:
    CapturePollThread (PollThread *poll)
        : Thread ("capture_poll")
        , _poll (poll)
    {}

protected:
    virtual bool loop () {
        XCamReturn ret = _poll->poll_buffer_loop ();

        if (ret == XCAM_RETURN_NO_ERROR || ret == XCAM_RETURN_ERROR_TIMEOUT)
            return true;
        return false;
    }

private:
    PollThread   *_poll;
};

const int PollThread::default_subdev_event_timeout = 100; // ms
const int PollThread::default_capture_event_timeout = 100; // ms

PollThread::PollThread ()
    : _poll_callback (NULL)
    , _stats_callback (NULL)
{
    _event_loop = new EventPollThread(this);
    _capture_loop = new CapturePollThread (this);

    XCAM_LOG_DEBUG ("PollThread constructed");
}

PollThread::~PollThread ()
{
    stop();

    XCAM_LOG_DEBUG ("~PollThread destructed");
}

bool
PollThread::set_capture_device (SmartPtr<V4l2Device> &dev)
{
    XCAM_ASSERT (!_capture_dev.ptr());
    _capture_dev = dev;
    return true;
}

bool
PollThread::set_event_device (SmartPtr<V4l2SubDevice> &dev)
{
    XCAM_ASSERT (!_event_dev.ptr());
    _event_dev = dev;
    return true;
}

bool
PollThread::set_poll_callback (PollCallback *callback)
{
    XCAM_ASSERT (!_poll_callback);
    _poll_callback = callback;
    return true;
}

bool
PollThread::set_stats_callback (StatsCallback *callback)
{
    XCAM_ASSERT (!_stats_callback);
    _stats_callback = callback;
    return true;
}

XCamReturn PollThread::start ()
{
    if (_event_dev.ptr () && !_event_loop->start ()) {
        return XCAM_RETURN_ERROR_THREAD;
    }
    if (!_capture_loop->start ()) {
        return XCAM_RETURN_ERROR_THREAD;
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn PollThread::stop ()
{
    _event_loop->stop ();
    _capture_loop->stop ();

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
PollThread::init_3a_stats_pool ()
{
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
PollThread::capture_3a_stats (SmartPtr<X3aStats> &stats)
{
    XCAM_UNUSED (stats);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
PollThread::handle_events (struct v4l2_event &event)
{
    XCAM_UNUSED (event);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
PollThread::handle_3a_stats_event (struct v4l2_event &event)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<X3aStats> stats;

    ret = capture_3a_stats (stats);
    if (ret != XCAM_RETURN_NO_ERROR || !stats.ptr()) {
        XCAM_LOG_WARNING ("capture 3a stats failed");
        return ret;
    }
    stats->set_timestamp (XCAM_TIMESPEC_2_USEC (event.timestamp));

    if (_stats_callback)
        return _stats_callback->x3a_stats_ready (stats);

    return ret;
}

XCamReturn
PollThread::poll_subdev_event_loop ()
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    struct v4l2_event event;
    int poll_ret = 0;

    poll_ret = _event_dev->poll_event (PollThread::default_subdev_event_timeout);

    if (poll_ret < 0) {
        XCAM_LOG_WARNING ("poll event failed but continue");
        ::usleep (100000); // 100ms
        return XCAM_RETURN_ERROR_TIMEOUT;
    }

    /* timeout */
    if (poll_ret == 0) {
        XCAM_LOG_DEBUG ("poll event timeout and continue");
        return XCAM_RETURN_ERROR_TIMEOUT;
    }

    xcam_mem_clear (event);
    ret = _event_dev->dequeue_event (event);
    if (ret != XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_WARNING ("dequeue event failed on dev:%s", XCAM_STR(_event_dev->get_device_name()));
        return XCAM_RETURN_ERROR_IOCTL;
    }

    ret = handle_events (event);
    return ret;
}

XCamReturn
PollThread::poll_buffer_loop ()
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    int poll_ret = 0;
    SmartPtr<V4l2Buffer> buf;

    poll_ret = _capture_dev->poll_event (PollThread::default_capture_event_timeout);

    if (poll_ret < 0) {
        XCAM_LOG_DEBUG ("poll buffer event got error but continue");
        ::usleep (100000); // 100ms
        return XCAM_RETURN_ERROR_TIMEOUT;
    }

    /* timeout */
    if (poll_ret == 0) {
        XCAM_LOG_DEBUG ("poll buffer timeout and continue");
        return XCAM_RETURN_ERROR_TIMEOUT;
    }

    ret = _capture_dev->dequeue_buffer (buf);
    if (ret != XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_WARNING ("capture buffer failed");
        return ret;
    }
    XCAM_ASSERT (buf.ptr());
    XCAM_ASSERT (_poll_callback);

    SmartPtr<VideoBuffer> video_buf = new V4l2BufferProxy (buf, _capture_dev);

    if (_poll_callback)
        return _poll_callback->poll_buffer_ready (video_buf);

    return ret;
}

};
