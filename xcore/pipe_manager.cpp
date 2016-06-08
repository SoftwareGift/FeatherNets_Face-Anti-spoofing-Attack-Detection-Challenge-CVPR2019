/*
 * pipe_manager.cpp - pipe manager
 *
 *  Copyright (c) 2016 Intel Corporation
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
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#include "pipe_manager.h"

#define XCAM_FAILED_STOP(exp, msg, ...)                 \
    if ((exp) != XCAM_RETURN_NO_ERROR) {                \
        XCAM_LOG_ERROR (msg, ## __VA_ARGS__);           \
        stop ();                                        \
        return ret;                                     \
    }

namespace XCam {

PipeManager::PipeManager ()
    : _is_running (false)
{
    _processor_center = new X3aImageProcessCenter;
    XCAM_LOG_DEBUG ("PipeManager construction");
}

PipeManager::~PipeManager ()
{
    XCAM_LOG_DEBUG ("PipeManager destruction");
}

bool
PipeManager::set_smart_analyzer (SmartPtr<SmartAnalyzer> analyzer)
{
    if (is_running ())
        return false;

    XCAM_ASSERT (analyzer.ptr () && !_smart_analyzer.ptr ());
    _smart_analyzer = analyzer;

    return true;
}

bool
PipeManager::add_image_processor (SmartPtr<ImageProcessor> processor)
{
    if (is_running ())
        return false;

    XCAM_ASSERT (processor.ptr ());
    return _processor_center->insert_processor (processor);
}

XCamReturn
PipeManager::start ()
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    if (_smart_analyzer.ptr ()) {
        if (_smart_analyzer->prepare_handlers () != XCAM_RETURN_NO_ERROR) {
            XCAM_LOG_INFO ("prepare smart analyzer handler failed");
        }

        _smart_analyzer->set_results_callback (this);
        if (_smart_analyzer->init (1920, 1080, 25) != XCAM_RETURN_NO_ERROR) {
            XCAM_LOG_INFO ("initialize smart analyzer failed");
        }
        if (_smart_analyzer->start () != XCAM_RETURN_NO_ERROR) {
            XCAM_LOG_INFO ("start smart analyzer failed");
        }
    }

    if (!_processor_center->has_processors ()) {
        XCAM_LOG_ERROR ("image processors empty");
    }
    _processor_center->set_image_callback (this);
    XCAM_FAILED_STOP (ret = _processor_center->start (), "3A process center start failed");

    _is_running = true;

    XCAM_LOG_DEBUG ("pipe manager started");
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
PipeManager::stop ()
{
    _is_running = false;

    if (_smart_analyzer.ptr ()) {
        _smart_analyzer->stop ();
        _smart_analyzer->deinit ();
    }

    if (_processor_center.ptr ())
        _processor_center->stop ();

    XCAM_LOG_DEBUG ("pipe manager stopped");
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
PipeManager::push_buffer (SmartPtr<VideoBuffer> &buf)
{
    // need to add sync mode later

    if (_processor_center->put_buffer (buf) == false) {
        XCAM_LOG_WARNING ("push buffer failed");
        return XCAM_RETURN_ERROR_UNKNOWN;
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
PipeManager::scaled_image_ready (const SmartPtr<BufferProxy> &buffer)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    if (!_smart_analyzer.ptr ()) {
        return XCAM_RETURN_NO_ERROR;
    }

    ret = _smart_analyzer->push_buffer (buffer);
    XCAM_FAIL_RETURN (ERROR,
                      ret == XCAM_RETURN_NO_ERROR,
                      ret,
                      "push scaled buffer failed");

    return XCAM_RETURN_NO_ERROR;
}

void
PipeManager::x3a_calculation_done (XAnalyzer *analyzer, X3aResultList &results)
{
    XCamReturn ret = _processor_center->put_3a_results (results);
    if (ret != XCAM_RETURN_NO_ERROR && ret != XCAM_RETURN_BYPASS) {
        XCAM_LOG_WARNING ("apply 3a results failed");
        return;
    }
    AnalyzerCallback::x3a_calculation_done (analyzer, results);
}

void
PipeManager::x3a_calculation_failed (XAnalyzer *analyzer, int64_t timestamp, const char *msg)
{
    AnalyzerCallback::x3a_calculation_failed (analyzer, timestamp, msg);
}

void
PipeManager::process_buffer_done (ImageProcessor *processor, const SmartPtr<VideoBuffer> &buf)
{
    ImageProcessCallback::process_buffer_done (processor, buf);
    post_buffer (buf);
}

void
PipeManager::process_buffer_failed (ImageProcessor *processor, const SmartPtr<VideoBuffer> &buf)
{
    ImageProcessCallback::process_buffer_failed (processor, buf);
}

void
PipeManager::process_image_result_done (ImageProcessor *processor, const SmartPtr<X3aResult> &result)
{
    ImageProcessCallback::process_image_result_done (processor, result);
}

};
