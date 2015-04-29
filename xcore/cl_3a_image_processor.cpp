/*
 * cl_3a_image_processor.cpp - CL 3A image processor
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
 */
#include "cl_3a_image_processor.h"
#include "cl_context.h"
#include "cl_blc_handler.h"
#include "cl_demosaic_handler.h"
#include "cl_csc_handler.h"
#include "cl_hdr_handler.h"
#include "cl_denoise_handler.h"
#include "cl_gamma_handler.h"
#include "cl_3a_stats_calculator.h"
#include "cl_wb_handler.h"
#include "cl_snr_handler.h"

namespace XCam {

CL3aImageProcessor::CL3aImageProcessor ()
    : CLImageProcessor ("CL3aImageProcessor")
    , _output_fourcc (V4L2_PIX_FMT_NV12)
    , _enable_hdr (false)
    , _enable_denoise (false)
    , _enable_gamma(false)
    , _out_smaple_type (OutSampleYuv)
    , _enable_snr (false)
{
    XCAM_LOG_DEBUG ("CL3aImageProcessor constructed");
}

CL3aImageProcessor::~CL3aImageProcessor ()
{
    XCAM_LOG_DEBUG ("CL3aImageProcessor destructed");
}

void
CL3aImageProcessor::set_stats_callback (const SmartPtr<StatsCallback> &callback)
{
    XCAM_ASSERT (callback.ptr ());
    _stats_callback = callback;
}

bool
CL3aImageProcessor::set_output_format (uint32_t fourcc)
{
    switch (fourcc) {
    case XCAM_PIX_FMT_RGBA64:
    case V4L2_PIX_FMT_XBGR32:
    case V4L2_PIX_FMT_ABGR32:
    case V4L2_PIX_FMT_BGR32:
    case V4L2_PIX_FMT_RGB32:
    case V4L2_PIX_FMT_ARGB32:
    case V4L2_PIX_FMT_XRGB32:
        _out_smaple_type = OutSampleRGB;
        break;
    case V4L2_PIX_FMT_NV12:
        _out_smaple_type = OutSampleYuv;
        break;
    default:
        XCAM_LOG_WARNING (
            "cl 3a processor doesn't support output format:%s",
            xcam_fourcc_to_string(fourcc));
        return false;
    }

    _output_fourcc = fourcc;
    return true;
}

bool
CL3aImageProcessor::can_process_result (SmartPtr<X3aResult> &result)
{
    XCAM_ASSERT (result.ptr());
    switch (result->get_type ()) {
    case XCAM_3A_RESULT_WHITE_BALANCE:
    case XCAM_3A_RESULT_BLACK_LEVEL:
    case XCAM_3A_RESULT_R_GAMMA:
    case XCAM_3A_RESULT_G_GAMMA:
    case XCAM_3A_RESULT_B_GAMMA:
        return true;

    default:
        return false;
    }

    return false;
}

XCamReturn
CL3aImageProcessor::apply_3a_results (X3aResultList &results)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    for (X3aResultList::iterator iter = results.begin (); iter != results.end (); ++iter)
    {
        SmartPtr<X3aResult> &result = *iter;
        ret = apply_3a_result (result);
        if (ret != XCAM_RETURN_NO_ERROR)
            break;
    }
    return ret;
}

XCamReturn
CL3aImageProcessor::apply_3a_result (SmartPtr<X3aResult> &result)
{
    STREAM_LOCK;

    uint32_t res_type = result->get_type ();

    switch (res_type) {
    case XCAM_3A_RESULT_WHITE_BALANCE: {
        SmartPtr<X3aWhiteBalanceResult> wb_res = result.dynamic_cast_ptr<X3aWhiteBalanceResult> ();
        XCAM_ASSERT (wb_res.ptr ());
        if (!_wb.ptr())
            break;
        _wb->set_wb_config (wb_res->get_standard_result ());
        break;
    }

    case XCAM_3A_RESULT_BLACK_LEVEL: {
        SmartPtr<X3aBlackLevelResult> bl_res = result.dynamic_cast_ptr<X3aBlackLevelResult> ();
        XCAM_ASSERT (bl_res.ptr ());
        if (!_black_level.ptr())
            break;
        //_black_level->set_bl_config (bl_res->get_standard_result ());
        break;
    }

    default:
        XCAM_LOG_WARNING ("CL3aImageProcessor unknow 3a result:%d", res_type);
        break;
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CL3aImageProcessor::create_handlers ()
{
    SmartPtr<CLImageHandler> image_handler;
    SmartPtr<CLContext> context = get_cl_context ();

    XCAM_ASSERT (context.ptr ());

    /* black leve as first */
    image_handler = create_cl_blc_image_handler (context);
    _black_level = image_handler;
    XCAM_FAIL_RETURN (
        WARNING,
        image_handler.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CL3aImageProcessor create blc handler failed");
    add_handler (image_handler);

    /* denoise */
    if (_enable_denoise) {
        image_handler = create_cl_denoise_image_handler (context);
        _denoise = image_handler;
        XCAM_FAIL_RETURN (
            WARNING,
            _denoise.ptr (),
            XCAM_RETURN_ERROR_CL,
            "CL3aImageProcessor create denoise handler failed");
        add_handler (image_handler);
    }

    image_handler = create_cl_3a_stats_image_handler (context);
    _x3a_stats_calculator = image_handler.dynamic_cast_ptr<CL3AStatsCalculator> ();
    XCAM_FAIL_RETURN (
        WARNING,
        _x3a_stats_calculator.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CL3aImageProcessor create 3a stats calculator failed");
    _x3a_stats_calculator->set_stats_callback (_stats_callback);
    add_handler (image_handler);

    image_handler = create_cl_wb_image_handler (context);
    _wb = image_handler.dynamic_cast_ptr<CLWbImageHandler> ();
    XCAM_FAIL_RETURN (
        WARNING,
        _wb.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CL3aImageProcessor create whitebalance handler failed");
    add_handler (image_handler);

    /* gamma */
    if(_enable_gamma) {
        image_handler = create_cl_gamma_image_handler (context);
        _gamma = image_handler.dynamic_cast_ptr<CLGammaImageHandler> ();
        XCAM_FAIL_RETURN (
            WARNING,
            _gamma.ptr (),
            XCAM_RETURN_ERROR_CL,
            "CL3aImageProcessor create gamma handler failed");
        add_handler (image_handler);
    }

    /* hdr */
    if (_enable_hdr) {
        image_handler = create_cl_hdr_image_handler (context, CL_HDR_TYPE_RGB);
        _hdr = image_handler;
        XCAM_FAIL_RETURN (
            WARNING,
            _hdr.ptr (),
            XCAM_RETURN_ERROR_CL,
            "CL3aImageProcessor create hdr handler failed");
        add_handler (image_handler);
    }

    /* demosaic */
    image_handler = create_cl_demosaic_image_handler (context);
    _demosaic = image_handler.dynamic_cast_ptr<CLBayer2RGBImageHandler> ();
    XCAM_FAIL_RETURN (
        WARNING,
        _demosaic.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CL3aImageProcessor create demosaic handler failed");
    add_handler (image_handler);

    /* simple noise reduction */
    if (_enable_snr) {
        image_handler = create_cl_snr_image_handler (context);
        _snr = image_handler;
        XCAM_FAIL_RETURN (
            WARNING,
            _snr.ptr (),
            XCAM_RETURN_ERROR_CL,
            "CL3aImageProcessor create snr handler failed");
        add_handler (image_handler);
    }

    /* color space conversion */
    if (_out_smaple_type == OutSampleYuv) {
        image_handler = create_cl_csc_image_handler (context, CL_CSC_TYPE_RGBATONV12);
        _csc = image_handler.dynamic_cast_ptr<CLCscImageHandler> ();
        XCAM_FAIL_RETURN (
            WARNING,
            _csc .ptr (),
            XCAM_RETURN_ERROR_CL,
            "CL3aImageProcessor create csc handler failed");
        add_handler (image_handler);
    } else if (_out_smaple_type == OutSampleRGB) {
        _demosaic->set_output_format (_output_fourcc);
    }

    return XCAM_RETURN_NO_ERROR;
}

};
