/*
 * device_manager.h - device manager
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

#ifndef XCAM_DEVICE_MANAGER_H
#define XCAM_DEVICE_MANAGER_H

#include "xcam_utils.h"
#include "smartptr.h"
#include "v4l2_device.h"
#include "v4l2_buffer_proxy.h"
#include "x3a_analyzer.h"
#include "smart_analyzer.h"
#include "x3a_image_process_center.h"
#include "image_processor.h"
#include "x3a_statistics_queue.h"
#include "poll_thread.h"
#include "stats_callback_interface.h"

namespace XCam {

enum XCamMessageType {
    XCAM_MESSAGE_BUF_OK = 0,
    XCAM_MESSAGE_BUF_ERROR,
    XCAM_MESSAGE_STATS_OK,
    XCAM_MESSAGE_STATS_ERROR,
    XCAM_MESSAGE_3A_RESULTS_OK,
    XCAM_MESSAGE_3A_RESULTS_ERROR,
};

struct XCamMessage {
    int64_t          timestamp;
    XCamMessageType  msg_id;
    char            *msg;

    XCamMessage (
        XCamMessageType type,
        int64_t timestamp = InvalidTimestamp,
        const char *message = NULL);
    ~XCamMessage ();

    XCAM_DEAD_COPY (XCamMessage);
};

class MessageThread;

class DeviceManager
    : public PollCallback
    , public StatsCallback
    , public AnalyzerCallback
    , public ImageProcessCallback
{
    friend class MessageThread;

public:
    DeviceManager();
    virtual ~DeviceManager();

    bool set_capture_device (SmartPtr<V4l2Device> device);
    bool set_event_device (SmartPtr<V4l2SubDevice> device);
    bool set_3a_analyzer (SmartPtr<X3aAnalyzer> analyzer);
    bool set_smart_analyzer (SmartPtr<SmartAnalyzer> analyzer);
    bool add_image_processor (SmartPtr<ImageProcessor> processor);
    bool set_poll_thread (SmartPtr<PollThread> thread);

    SmartPtr<V4l2Device>& get_capture_device () {
        return _device;
    }
    SmartPtr<V4l2SubDevice>& get_event_device () {
        return _subdevice;
    }
    SmartPtr<X3aAnalyzer>& get_analyzer () {
        return _3a_analyzer;
    }

    bool is_running () const {
        return _is_running;
    }
    bool has_3a () const {
        return _has_3a;
    }

    XCamReturn start ();
    XCamReturn stop ();

protected:
    virtual void handle_message (const SmartPtr<XCamMessage> &msg) = 0;
    virtual void handle_buffer (const SmartPtr<VideoBuffer> &buf) = 0;

protected:
    //virtual functions derived from PollCallback
    virtual XCamReturn poll_buffer_ready (SmartPtr<VideoBuffer> &buf);
    virtual XCamReturn poll_buffer_failed (int64_t timestamp, const char *msg);
    virtual XCamReturn x3a_stats_ready (const SmartPtr<X3aStats> &stats);
    virtual XCamReturn dvs_stats_ready ();
    virtual XCamReturn scaled_image_ready (const SmartPtr<BufferProxy> &buffer);

    //virtual functions derived from AnalyzerCallback
    virtual void x3a_calculation_done (XAnalyzer *analyzer, X3aResultList &results);
    virtual void x3a_calculation_failed (XAnalyzer *analyzer, int64_t timestamp, const char *msg);

    //virtual functions derived from ImageProcessCallback
    virtual void process_buffer_done (ImageProcessor *processor, const SmartPtr<VideoBuffer> &buf);
    virtual void process_buffer_failed (ImageProcessor *processor, const SmartPtr<VideoBuffer> &buf);
    virtual void process_image_result_done (ImageProcessor *processor, const SmartPtr<X3aResult> &result);

private:
    void post_message (XCamMessageType type, int64_t timestamp, const char *msg);
    XCamReturn message_loop ();

    XCAM_DEAD_COPY (DeviceManager);

protected:
    SmartPtr<V4l2Device>             _device;
    SmartPtr<V4l2SubDevice>          _subdevice;
    SmartPtr<PollThread>             _poll_thread;

    /* 3A calculation and image processing*/
    bool                             _has_3a;
    SmartPtr<X3aAnalyzer>            _3a_analyzer;
    SmartPtr<X3aImageProcessCenter>  _3a_process_center;

    /* msg queue */
    SafeList<XCamMessage>            _msg_queue;
    SmartPtr<MessageThread>          _msg_thread;

    bool                             _is_running;

    /* smart analysis */
    SmartPtr<SmartAnalyzer>         _smart_analyzer;
};

};

#endif //XCAM_DEVICE_MANAGER_H
