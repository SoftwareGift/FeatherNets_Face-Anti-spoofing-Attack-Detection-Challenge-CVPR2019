/*
 * x3a_ciq_tuning_handler.cpp - x3a Common IQ tuning handler
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
 * Author: Zong Wei <wei.zong@intel.com>
 */

#include "x3a_analyzer.h"
#include "x3a_ciq_tuning_handler.h"

#define X3A_CIQ_CAMERA_ID   "IMX185"

namespace XCam {

X3aCiqTuningHandler::X3aCiqTuningHandler (const char *name)
    : _tuning_data (NULL)
    , _name (NULL)
{
    if (name)
        _name = strndup (name, XCAM_MAX_STR_SIZE);
}

X3aCiqTuningHandler::~X3aCiqTuningHandler ()
{
    if (_name)
        xcam_free (_name);
}

void
X3aCiqTuningHandler::set_tuning_data (void* data)
{
    if (NULL != data) {
        _tuning_data = data;
    }
}

void
X3aCiqTuningHandler::set_ae_handler (SmartPtr<AeHandler> &handler)
{
    if (!_ae_handler.ptr ()) {
        _ae_handler = handler;
    }
}

void
X3aCiqTuningHandler::set_awb_handler (SmartPtr<AwbHandler> &handler)
{
    if (!_awb_handler.ptr ()) {
        _awb_handler = handler;
    }
}

double
X3aCiqTuningHandler::get_max_analog_gain()
{
    AnalyzerHandler::HandlerLock lock(this);

    if (_ae_handler.ptr ()) {
        return _ae_handler->get_max_analog_gain ();
    }
    return 0.0;
}

double
X3aCiqTuningHandler::get_current_analog_gain ()
{
    AnalyzerHandler::HandlerLock lock(this);

    if (_ae_handler.ptr ()) {
        return _ae_handler->get_current_analog_gain ();
    }
    return 0.0;
}

int64_t
X3aCiqTuningHandler::get_current_exposure_time ()
{
    AnalyzerHandler::HandlerLock lock(this);

    if (_ae_handler.ptr ()) {
        return _ae_handler->get_current_exposure_time ();
    }
    return 0.0;
}

uint32_t
X3aCiqTuningHandler::get_current_estimate_cct ()
{
    AnalyzerHandler::HandlerLock lock(this);

    if (_awb_handler.ptr ()) {
        return _awb_handler->get_current_estimate_cct ();
    }
    return 0;
}

};
