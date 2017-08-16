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
#include "smart_analyzer.h"
#include "smart_analysis_handler.h"

#include "xcam_obj_debug.h"

namespace XCam {

SmartAnalyzer::SmartAnalyzer (const char *name)
    : XAnalyzer (name)
{
    XCAM_OBJ_PROFILING_INIT;
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
    handler->set_analyzer (this);
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
    SmartHandlerList::iterator i_handler = _handlers.begin ();
    for (; i_handler != _handlers.end ();  ++i_handler)
    {
        SmartPtr<SmartAnalysisHandler> handler = *i_handler;
        XCamReturn ret = handler->create_context (handler);
        if (ret != XCAM_RETURN_NO_ERROR) {
            XCAM_LOG_WARNING ("smart analyzer initialize handler(%s) context failed", XCAM_STR(handler->get_name()));
        }
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
SmartAnalyzer::internal_deinit ()
{
    SmartHandlerList::iterator i_handler = _handlers.begin ();
    for (; i_handler != _handlers.end ();  ++i_handler)
    {
        SmartPtr<SmartAnalysisHandler> handler = *i_handler;
        if (handler->is_valid ())
            handler->destroy_context ();
    }

    return XCAM_RETURN_NO_ERROR;
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

    SmartHandlerList::iterator i_handler = _handlers.begin ();
    for (; i_handler != _handlers.end ();  ++i_handler)
    {
        SmartPtr<SmartAnalysisHandler> handler = *i_handler;
        if (!handler->is_valid ())
            continue;

        ret = handler->update_params (params);

        if (ret != XCAM_RETURN_NO_ERROR) {
            XCAM_LOG_WARNING ("smart analyzer update handler(%s) context failed", XCAM_STR(handler->get_name()));
            handler->destroy_context ();
        }
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
SmartAnalyzer::analyze (const SmartPtr<VideoBuffer> &buffer)
{
    XCAM_OBJ_PROFILING_START;

    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    X3aResultList results;

    if (!buffer.ptr ()) {
        XCAM_LOG_DEBUG ("SmartAnalyzer::analyze got NULL buffer!");
        return XCAM_RETURN_ERROR_PARAM;
    }

    SmartHandlerList::iterator i_handler = _handlers.begin ();
    for (; i_handler != _handlers.end ();  ++i_handler)
    {
        SmartPtr<SmartAnalysisHandler> handler = *i_handler;
        if (!handler->is_valid ())
            continue;

        ret = handler->analyze (buffer, results);
        if (ret != XCAM_RETURN_NO_ERROR && ret != XCAM_RETURN_BYPASS) {
            XCAM_LOG_WARNING ("smart analyzer analyze handler(%s) context failed", XCAM_STR(handler->get_name()));
            handler->destroy_context ();
        }
    }

    if (!results.empty ()) {
        set_results_timestamp (results, buffer->get_timestamp ());
        notify_calculation_done (results);
    }

    XCAM_OBJ_PROFILING_END ("smart analysis", XCAM_OBJ_DUR_FRAME_NUM);

    return XCAM_RETURN_NO_ERROR;
}

void
SmartAnalyzer::post_smart_results (X3aResultList &results, int64_t timestamp)
{
    if (!results.empty ()) {
        set_results_timestamp (results, timestamp);
        notify_calculation_done (results);
    }
}

}
