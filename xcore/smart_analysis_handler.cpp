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
 */

#include "smart_analysis_handler.h"
#include "buffer_pool.h"

namespace XCam {

SmartAnalysisHandler::SmartAnalysisHandler (XCamSmartAnalysisDescription *desc, SmartPtr<SmartAnalyzerLoader> &loader, const char *name)
    : _desc (desc)
    , _loader (loader)
    , _name (NULL)
    , _context (NULL)
{
    if (name)
        _name = strndup (name, XCAM_MAX_STR_SIZE);

    create_context ();
}

SmartAnalysisHandler::~SmartAnalysisHandler ()
{
    if (_name)
        xcam_free (_name);

    destroy_context ();
}

XCamReturn
SmartAnalysisHandler::create_context ()
{
    XCamSmartAnalysisContext *context = NULL;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    XCAM_ASSERT (!_context);
    if ((ret = _desc->create_context (&context)) != XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_WARNING ("smart handler(%s) lib create context failed", XCAM_STR(get_name()));
        return ret;
    }
    _context = context;
    return XCAM_RETURN_NO_ERROR;
}

void
SmartAnalysisHandler::destroy_context ()
{
    if (_context && _desc && _desc->destroy_context) {
        _desc->destroy_context (_context);
        _context = NULL;
    }
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
SmartAnalysisHandler::analyze (XCamVideoBuffer *buffer, X3aResultList &results)
{
    XCAM_LOG_DEBUG ("smart handler(%s) analyze", XCAM_STR(get_name()));
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    XCAM_ASSERT (_context);

    ret = _desc->analyze (_context, buffer);
    XCAM_FAIL_RETURN (WARNING,
                      ret == XCAM_RETURN_NO_ERROR,
                      ret,
                      "smart handler(%s) calculation failed", XCAM_STR(get_name()));

    XCam3aResultHead *res_array[XCAM_3A_MAX_RESULT_COUNT];
    uint32_t res_count = 0;

    xcam_mem_clear (res_array);
    XCAM_ASSERT (_context);
    ret = _desc->get_results (_context, res_array, &res_count);
    XCAM_FAIL_RETURN (WARNING,
                      ret == XCAM_RETURN_NO_ERROR,
                      ret,
                      "samrt handler(%s) get results failed", XCAM_STR(get_name()));

    if (res_count) {
        ret = convert_results (res_array, res_count, results);
        XCAM_FAIL_RETURN (WARNING,
                          ret == XCAM_RETURN_NO_ERROR,
                          ret,
                          "smart handler(%s) convert_results failed", XCAM_STR(get_name()));
        _desc->free_results (res_array, res_count);
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
