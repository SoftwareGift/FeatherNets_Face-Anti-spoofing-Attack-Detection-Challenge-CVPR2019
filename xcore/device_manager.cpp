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

#include "device_manager.h"
#include "poll_thread.h"
#include "xcam_thread.h"
#include "x3a_image_process_center.h"
#include "x3a_analyzer_manager.h"
#include "isp_image_processor.h"
#include "isp_controller.h"
#if HAVE_IA_AIQ
#include "x3a_analyzer_aiq.h"
#endif

#define XCAM_FAILED_STOP(exp, msg, ...)                 \
    if ((exp) != XCAM_RETURN_NO_ERROR) {                \
        XCAM_LOG_ERROR (msg, ## __VA_ARGS__);           \
        stop ();                                        \
        return ret;                                     \
    }

namespace XCam {

class MessageThread
    : public Thread
{
public:
    explicit MessageThread (DeviceManager *dev_manager)
        : Thread ("MessageThread")
        , _manager (dev_manager)
    {}

protected:
    virtual bool loop ();

    DeviceManager *_manager;
};

bool
MessageThread::loop()
{
    XCamReturn ret = _manager->message_loop();
    if (ret == XCAM_RETURN_NO_ERROR || ret == XCAM_RETURN_ERROR_TIMEOUT)
        return true;

    return false;
}

XCamMessage::XCamMessage (XCamMessageType type, int64_t timestamp, const char *message)
    : timestamp (timestamp)
    , msg_id (type)
{
    if (message)
        this->msg = strdup (message);
}

XCamMessage::~XCamMessage ()
{
    if (msg)
        xcam_free (msg);
}

DeviceManager::DeviceManager()
    : _has_3a (true)
    , _is_running (false)
{
    _3a_process_center = new X3aImageProcessCenter;
    XCAM_LOG_DEBUG ("~DeviceManager construction");
}

DeviceManager::~DeviceManager()
{
    XCAM_LOG_DEBUG ("~DeviceManager destruction");
}

bool
DeviceManager::set_capture_device (SmartPtr<V4l2Device> device)
{
    if (is_running())
        return false;

    XCAM_ASSERT (device.ptr () && !_device.ptr ());
    _device = device;
    return true;
}
bool
DeviceManager::set_event_device (SmartPtr<V4l2SubDevice> device)
{
    if (is_running())
        return false;

    XCAM_ASSERT (device.ptr () && !_subdevice.ptr ());
    _subdevice = device;
    return true;
}

bool
DeviceManager::set_isp_controller (SmartPtr<IspController> controller)
{
    if (is_running())
        return false;

    XCAM_ASSERT (controller.ptr () && !_isp_controller.ptr ());
    _isp_controller = controller;
    return true;
}

bool
DeviceManager::set_analyzer (SmartPtr<X3aAnalyzer> analyzer)
{
    if (is_running())
        return false;

    XCAM_ASSERT (analyzer.ptr () && !_3a_analyzer.ptr ());
    _3a_analyzer = analyzer;
    return true;
}

bool
DeviceManager::add_image_processor (SmartPtr<ImageProcessor> processor)
{
    if (is_running())
        return false;

    XCAM_ASSERT (processor.ptr ());
    return _3a_process_center->insert_processor (processor);
}

XCamReturn
DeviceManager::start ()
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    // start device
    XCAM_ASSERT (_device->is_opened());
    if (!_device.ptr() || !_device->is_opened()) {
        XCAM_FAILED_STOP (ret = XCAM_RETURN_ERROR_FILE, "capture device not ready");
    }
    XCAM_FAILED_STOP (ret = _device->start(), "capture device start failed");

    //start subdevice
    XCAM_ASSERT (_subdevice->is_opened());
    if (!_subdevice.ptr() || !_subdevice->is_opened()) {
        XCAM_FAILED_STOP (ret = XCAM_RETURN_ERROR_FILE, "event device not ready");
    }
    XCAM_FAILED_STOP (ret = _subdevice->start(), "start event device failed");

    //suppose _device and _subdevice already started
    if (!_isp_controller.ptr ())
        _isp_controller = new IspController (_device);
    XCAM_ASSERT (_isp_controller.ptr());

    if (_has_3a) {
        // Initialize and start analyzer
        uint32_t width = 0, height = 0;
        uint32_t fps_n = 0, fps_d = 0;
        double framerate = 30.0;

        if (!_3a_analyzer.ptr()) {
            _3a_analyzer = X3aAnalyzerManager::instance()->create_analyzer();
        }
        if (!_3a_analyzer.ptr()) {
            XCAM_FAILED_STOP (ret = XCAM_RETURN_ERROR_PARAM, "create analyzer failed");
        }
        _3a_analyzer->set_results_callback (this);

        _device->get_size (width, height);
        _device->get_framerate (fps_n, fps_d);
        if (fps_d)
            framerate = (double)fps_n / (double)fps_d;
        XCAM_FAILED_STOP (
            ret = _3a_analyzer->init (width, height, framerate),
            "initialize analyzer failed");

        XCAM_FAILED_STOP (ret = _3a_analyzer->start (), "start analyzer failed");

        // Initialize and start image processors
        if (!_3a_process_center->has_processors()) {
            // default processor
            SmartPtr<ImageProcessor> default_processor = new IspImageProcessor (_isp_controller);
            XCAM_ASSERT (default_processor.ptr ());
            _3a_process_center->insert_processor (default_processor);
        }

        _3a_process_center->set_image_callback(this);
        XCAM_FAILED_STOP (ret = _3a_process_center->start (), "3A process center start failed");

    }

    //Initialize and start poll thread
    _poll_thread = new PollThread;
    _poll_thread->set_capture_device (_device);
    _poll_thread->set_event_device (_subdevice);
    _poll_thread->set_isp_controller (_isp_controller);
    _poll_thread->set_callback (this);

    XCAM_FAILED_STOP (ret = _poll_thread->start(), "start poll failed");

    _is_running = true;

    XCAM_LOG_DEBUG ("Device manager started");
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DeviceManager::stop ()
{
    _is_running = false;

    if (_poll_thread.ptr())
        _poll_thread->stop ();

    if (_3a_analyzer.ptr()) {
        _3a_analyzer->stop ();
        _3a_analyzer->deinit ();
    }

    if (_3a_process_center.ptr())
        _3a_process_center->stop ();

    _subdevice->stop ();
    _device->stop ();

    _isp_controller.release ();
    _poll_thread.release ();

    XCAM_LOG_DEBUG ("Device manager stopped");
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DeviceManager::poll_3a_stats_ready (SmartPtr<X3aStats> &stats)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    X3aResultList results;
    XCAM_ASSERT (_3a_analyzer.ptr());

    ret = _3a_analyzer->push_3a_stats (stats);
    XCAM_FAIL_RETURN (ERROR,
                      ret == XCAM_RETURN_NO_ERROR,
                      ret,
                      "analyze 3a statistics failed");

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DeviceManager::poll_dvs_stats_ready ()
{
    XCAM_ASSERT (false);
    // TODO
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DeviceManager::poll_buffer_ready (SmartPtr<V4l2BufferProxy> &buf)
{
    if (_has_3a) {
        SmartPtr<VideoBuffer> video_buf = buf;
        if (_3a_process_center->put_buffer (video_buf) == false)
            return XCAM_RETURN_ERROR_UNKNOWN;
    }
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DeviceManager::poll_buffer_failed (int64_t timestamp, const char *msg)
{
    post_message (XCAM_MESSAGE_BUF_ERROR, timestamp, msg);
    return XCAM_RETURN_NO_ERROR;
}

void
DeviceManager::x3a_calculation_done (X3aAnalyzer *analyzer, X3aResultList &results)
{
    XCamReturn ret = _3a_process_center->put_3a_results (results);
    if (ret != XCAM_RETURN_NO_ERROR && ret != XCAM_RETURN_BYPASS) {
        XCAM_LOG_WARNING ("apply 3a results failed");
        return;
    }
    AnalyzerCallback::x3a_calculation_done (analyzer, results);
}

void
DeviceManager::x3a_calculation_failed (X3aAnalyzer *analyzer, int64_t timestamp, const char *msg)
{
    AnalyzerCallback::x3a_calculation_failed (analyzer, timestamp, msg);
}

void
DeviceManager::process_buffer_done (ImageProcessor *processor, SmartPtr<VideoBuffer> &buf)
{
    ImageProcessCallback::process_buffer_done (processor, buf);
    handle_buffer (buf);
}

void
DeviceManager::process_buffer_failed (ImageProcessor *processor, SmartPtr<VideoBuffer> &buf)
{
    ImageProcessCallback::process_buffer_failed (processor, buf);
}

void
DeviceManager::process_image_result_done (ImageProcessor *processor, SmartPtr<X3aResult> &result)
{
    ImageProcessCallback::process_image_result_done (processor, result);
}

void
DeviceManager::post_message (XCamMessageType type, int64_t timestamp, const char *msg)
{
    SmartPtr<XCamMessage> new_msg = new XCamMessage (type, timestamp, msg);
    _msg_queue.push (new_msg);
}

XCamReturn
DeviceManager::message_loop ()
{
    const static int32_t msg_time_out = -1; //wait until wakeup
    SmartPtr<XCamMessage> msg = _msg_queue.pop (msg_time_out);
    if (!msg.ptr ())
        return XCAM_RETURN_ERROR_THREAD;
    handle_message (msg);
    return XCAM_RETURN_NO_ERROR;
}

};
