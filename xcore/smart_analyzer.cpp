/*
 * smart_analyzer.cpp - smart analyzer
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
 * Author: Zong Wei <wei.zong@intel.com>
 */

#include "smart_analyzer_loader.h"
#include "scaled_buffer_pool.h"
#include "smart_analyzer.h"
#include "smart_analysis_handler.h"

namespace XCam {

SmartAnalyzer::SmartAnalyzer (const char *name)
    : XAnalyzer (name)
{
}

SmartAnalyzer::SmartAnalyzer (SmartPtr<SmartAnalysisHandler> handler, const char *name)
    : XAnalyzer (name)
{
    if (!handler.ptr ())
        add_handler (handler);
}

SmartAnalyzer::~SmartAnalyzer ()
{
}

XCamReturn
SmartAnalyzer::add_handler (SmartPtr<SmartAnalysisHandler> handler)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    if (!handler.ptr ()) {
        return XCAM_RETURN_ERROR_PARAM;
    }

    _handlers.push_back (handler);
    return ret;
}

XCamReturn
SmartAnalyzer::create_handlers ()
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    if (_handlers.empty ()) {
        ret = XCAM_RETURN_ERROR_PARAM;
    }
    return ret;
}

XCamReturn
SmartAnalyzer::release_handlers ()
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    return ret;
}

XCamReturn
SmartAnalyzer::internal_init (uint32_t width, uint32_t height, double framerate)
{
    XCAM_UNUSED (width);
    XCAM_UNUSED (height);
    XCAM_UNUSED (framerate);
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    return ret;
}

XCamReturn
SmartAnalyzer::internal_deinit ()
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    return ret;
}

XCamReturn
SmartAnalyzer::configure ()
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    return ret;
}

XCamReturn
SmartAnalyzer::update_params (XCamSmartAnalysisParam &params)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    SamrtAnalysisHandlerList::iterator i_handler = _handlers.begin ();
    for (; i_handler != _handlers.end ();  ++i_handler)
    {
        ret = (*i_handler)->update_params (params);
    }

    XCAM_FAIL_RETURN (WARNING,
                      ret == XCAM_RETURN_NO_ERROR,
                      ret,
                      "smart analyzer update parameters failed");

    return ret;
}

XCamReturn
SmartAnalyzer::analyze (SmartPtr<BufferProxy> &buffer)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    X3aResultList results;
    XCamVideoBuffer videoBuffer;

    if (!buffer.ptr ()) {
        XCAM_LOG_DEBUG ("SmartAnalyzer::analyze got NULL buffer !");
        return XCAM_RETURN_ERROR_PARAM;
    }

    SmartPtr<ScaledVideoBuffer> scaledBuffer = buffer.dynamic_cast_ptr<ScaledVideoBuffer> ();
    scaledBuffer->get_scaled_buffer (videoBuffer);

    SamrtAnalysisHandlerList::iterator i_handler = _handlers.begin ();
    for (; i_handler != _handlers.end ();  ++i_handler)
    {
        ret = (*i_handler)->analyze (&videoBuffer, results);
    }

    XCAM_FAIL_RETURN (WARNING,
                      ret == XCAM_RETURN_NO_ERROR,
                      ret,
                      "smart analyzer calculation failed");

    if (!results.empty ()) {
        set_results_timestamp(results, buffer->get_timestamp ());
        notify_calculation_done (results);
    }

    return ret;
}

}
