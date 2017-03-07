/*
 * smart_analysis_handler.cpp - smart analysis handler
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
 *             Wind Yuan <feng.yuan@intel.com>
 */

#include "smart_analysis_handler.h"
#include "smart_analyzer_loader.h"
#include "smart_analyzer.h"
#include "drm_bo_buffer.h"
#include "buffer_pool.h"

namespace XCam {

SmartAnalysisHandler::SmartHandlerMap SmartAnalysisHandler::_handler_map;
Mutex SmartAnalysisHandler::_handler_map_lock;

SmartAnalysisHandler::SmartAnalysisHandler (XCamSmartAnalysisDescription *desc, SmartPtr<SmartAnalyzerLoader> &loader, const char *name)
    : _desc (desc)
    , _loader (loader)
    , _analyzer (NULL)
    , _name (NULL)
    , _context (NULL)
    , _async_mode (false)
{
    if (name)
        _name = strndup (name, XCAM_MAX_STR_SIZE);
}

SmartAnalysisHandler::~SmartAnalysisHandler ()
{
    if (is_valid ())
        destroy_context ();

    if (_name)
        xcam_free (_name);
}

XCamReturn
SmartAnalysisHandler::create_context (SmartPtr<SmartAnalysisHandler> &self)
{
    XCamSmartAnalysisContext *context = NULL;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    uint32_t async_mode = 0;
    XCAM_ASSERT (!_context);
    XCAM_ASSERT (self.ptr () == this);
    if ((ret = _desc->create_context (&context, &async_mode, NULL)) != XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_WARNING ("smart handler(%s) lib create context failed", XCAM_STR(get_name()));
        return ret;
    }
    if (!context) {
        XCAM_LOG_WARNING ("smart handler(%s) lib create context failed with NULL context", XCAM_STR(get_name()));
        return XCAM_RETURN_ERROR_UNKNOWN;
    }
    _async_mode = async_mode;

    XCAM_LOG_INFO ("create smart analysis context(%s)", XCAM_STR(get_name()));

    SmartLock locker (_handler_map_lock);
    _handler_map[context] = self;
    _context = context;
    return XCAM_RETURN_NO_ERROR;
}

void
SmartAnalysisHandler::destroy_context ()
{
    XCamSmartAnalysisContext *context;
    {
        SmartLock locker (_handler_map_lock);
        context = _context;
        _context = NULL;
        if (context)
            _handler_map.erase (context);
    }

    if (context && _desc && _desc->destroy_context) {
        _desc->destroy_context (context);
        XCAM_LOG_INFO ("destroy smart analysis context(%s)", XCAM_STR(get_name()));
    }
}

XCamReturn
SmartAnalysisHandler::post_aync_results (
    XCamSmartAnalysisContext *context,
    const XCamVideoBuffer *buffer,
    XCam3aResultHead *results[], uint32_t res_count)
{
    SmartPtr<SmartAnalysisHandler> handler = NULL;
    XCAM_ASSERT (context);
    {
        SmartLock locker (_handler_map_lock);
        SmartHandlerMap::iterator i_h = _handler_map.find (context);
        if (i_h != _handler_map.end ())
            handler = i_h->second;
    }

    if (!handler.ptr ()) {
        XCAM_LOG_WARNING ("can't find a proper smart analyzer handler, please check context pointer");
        return XCAM_RETURN_ERROR_PARAM;
    }

    return handler->post_smart_results (buffer, results, res_count);
}

XCamReturn
SmartAnalysisHandler::post_smart_results (const XCamVideoBuffer *buffer, XCam3aResultHead *results[], uint32_t res_count)
{
    X3aResultList result_list;
    XCamReturn ret = convert_results (results, res_count, result_list);
    XCAM_FAIL_RETURN (
        WARNING,
        ret == XCAM_RETURN_NO_ERROR,
        ret,
        "smart handler convert results failed in async mode");

    if (_analyzer)
        _analyzer->post_smart_results (result_list, (buffer ? buffer->timestamp : InvalidTimestamp));

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
SmartAnalysisHandler::update_params (XCamSmartAnalysisParam &params)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    XCAM_ASSERT (_context);
    ret = _desc->update_params (_context, &params);
    XCAM_FAIL_RETURN (WARNING,
                      ret == XCAM_RETURN_NO_ERROR,
                      ret,
                      "smart handler(%s) update parameters failed", XCAM_STR(get_name()));

    return ret;
}

XCamReturn
SmartAnalysisHandler::analyze (SmartPtr<BufferProxy> &buffer, X3aResultList &results)
{
    XCAM_LOG_DEBUG ("smart handler(%s) analyze", XCAM_STR(get_name()));
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    XCamVideoBuffer *video_buffer = convert_to_external_buffer (buffer);
    XCam3aResultHead *res_array[XCAM_3A_MAX_RESULT_COUNT];
    uint32_t res_count = XCAM_3A_MAX_RESULT_COUNT;

    XCAM_ASSERT (buffer.ptr ());
    XCAM_ASSERT (_context);
    XCAM_ASSERT (video_buffer);
    xcam_mem_clear (res_array);

    ret = _desc->analyze (_context, video_buffer, res_array, &res_count);
    XCAM_ASSERT (video_buffer->unref);
    video_buffer->unref (video_buffer);
    XCAM_FAIL_RETURN (WARNING,
                      ret == XCAM_RETURN_NO_ERROR,
                      ret,
                      "smart handler(%s) calculation failed", XCAM_STR(get_name()));

    if (res_count > 0 && res_array[0]) {
        ret = convert_results (res_array, res_count, results);
        XCAM_FAIL_RETURN (WARNING,
                          ret == XCAM_RETURN_NO_ERROR,
                          ret,
                          "smart handler(%s) convert_results failed", XCAM_STR(get_name()));
        _desc->free_results (_context, res_array, res_count);
    }

    return ret;
}

XCamReturn
SmartAnalysisHandler::convert_results (XCam3aResultHead *from[], uint32_t from_count, X3aResultList &to)
{
    for (uint32_t i = 0; i < from_count; ++i) {
        SmartPtr<X3aResult> standard_res =
            X3aResultFactory::instance ()->create_3a_result (from[i]);
        to.push_back (standard_res);
    }

    return XCAM_RETURN_NO_ERROR;
}

}
