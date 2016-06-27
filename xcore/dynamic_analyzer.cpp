/*
 * analyzer_loader.cpp - analyzer loader
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
 * Author: Wind Yuan <feng.yuan@intel.com>
 *         Jia Meng <jia.meng@intel.com>
 */

#include "dynamic_analyzer.h"

namespace XCam {

DynamicAnalyzer::DynamicAnalyzer (XCam3ADescription *desc, SmartPtr<AnalyzerLoader> &loader, const char *name)
    : X3aAnalyzer (name)
    , _desc (desc)
    , _context (NULL)
    , _loader (loader)
{
}

DynamicAnalyzer::~DynamicAnalyzer ()
{
    destroy_context ();
}

XCamReturn
DynamicAnalyzer::create_context ()
{
    XCam3AContext *context = NULL;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    XCAM_ASSERT (!_context);
    if ((ret = _desc->create_context (&context)) != XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_WARNING ("dynamic 3a lib create context failed");
        return ret;
    }
    _context = context;
    return XCAM_RETURN_NO_ERROR;
}

void
DynamicAnalyzer::destroy_context ()
{
    if (_context && _desc && _desc->destroy_context) {
        _desc->destroy_context (_context);
        _context = NULL;
    }
}

XCamReturn
DynamicAnalyzer::analyze_ae (XCamAeParam &param)
{
    XCAM_ASSERT (_context);
    return _desc->analyze_ae (_context, &param);
}

XCamReturn
DynamicAnalyzer::analyze_awb (XCamAwbParam &param)
{
    XCAM_ASSERT (_context);
    return _desc->analyze_awb (_context, &param);
}

XCamReturn
DynamicAnalyzer::analyze_af (XCamAfParam &param)
{
    XCAM_ASSERT (_context);
    return _desc->analyze_af (_context, &param);
}

SmartPtr<AeHandler>
DynamicAnalyzer::create_ae_handler ()
{
    return new DynamicAeHandler (this);
}

SmartPtr<AwbHandler>
DynamicAnalyzer::create_awb_handler ()
{
    return new DynamicAwbHandler (this);
}

SmartPtr<AfHandler>
DynamicAnalyzer::create_af_handler ()
{
    return new DynamicAfHandler (this);
}

SmartPtr<CommonHandler>
DynamicAnalyzer::create_common_handler ()
{
    if (_common_handler.ptr())
        return _common_handler;

    _common_handler = new DynamicCommonHandler (this);
    return _common_handler;
}

XCamReturn
DynamicAnalyzer::internal_init (uint32_t width, uint32_t height, double framerate)
{
    XCAM_UNUSED (width);
    XCAM_UNUSED (height);
    XCAM_UNUSED (framerate);
    return create_context ();
}

XCamReturn
DynamicAnalyzer::internal_deinit ()
{
    destroy_context ();
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DynamicAnalyzer::configure_3a ()
{
    uint32_t width = get_width ();
    uint32_t height = get_height ();
    double framerate = get_framerate ();
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    XCAM_ASSERT (_context);

    ret = _desc->configure_3a (_context, width, height, framerate);
    XCAM_FAIL_RETURN (WARNING,
                      ret == XCAM_RETURN_NO_ERROR,
                      ret,
                      "dynamic analyzer configure 3a failed");
    set_manual_brightness(_brightness_level_param);

    return XCAM_RETURN_NO_ERROR;
}
XCamReturn
DynamicAnalyzer::pre_3a_analyze (SmartPtr<X3aStats> &stats)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    XCamCommonParam common_params = _common_handler->get_params_unlock ();

    XCAM_ASSERT (_context);
    _cur_stats = stats;
    ret = _desc->set_3a_stats (_context, stats->get_stats (), stats->get_timestamp ());
    XCAM_FAIL_RETURN (WARNING,
                      ret == XCAM_RETURN_NO_ERROR,
                      ret,
                      "dynamic analyzer set_3a_stats failed");

    ret = _desc->update_common_params (_context, &common_params);
    XCAM_FAIL_RETURN (WARNING,
                      ret == XCAM_RETURN_NO_ERROR,
                      ret,
                      "dynamic analyzer update common params failed");

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
DynamicAnalyzer::post_3a_analyze (X3aResultList &results)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    XCam3aResultHead *res_array[XCAM_3A_MAX_RESULT_COUNT];
    uint32_t res_count = XCAM_3A_MAX_RESULT_COUNT;

    xcam_mem_clear (res_array);
    XCAM_ASSERT (_context);
    ret = _desc->combine_analyze_results (_context, res_array, &res_count);
    XCAM_FAIL_RETURN (WARNING,
                      ret == XCAM_RETURN_NO_ERROR,
                      ret,
                      "dynamic analyzer combine_analyze_results failed");

    _cur_stats.release ();

    if (res_count) {
        ret = convert_results (res_array, res_count, results);
        XCAM_FAIL_RETURN (WARNING,
                          ret == XCAM_RETURN_NO_ERROR,
                          ret,
                          "dynamic analyzer convert_results failed");
        _desc->free_results (res_array, res_count);
    }

    return XCAM_RETURN_NO_ERROR;
}

const XCamCommonParam
DynamicAnalyzer::get_common_params ()
{
    return _common_handler->get_params_unlock ();
}

XCamReturn
DynamicAnalyzer::convert_results (XCam3aResultHead *from[], uint32_t from_count, X3aResultList &to)
{
    for (uint32_t i = 0; i < from_count; ++i) {
        SmartPtr<X3aResult> standard_res =
            X3aResultFactory::instance ()->create_3a_result (from[i]);
        to.push_back (standard_res);
    }

    return XCAM_RETURN_NO_ERROR;
}
}
