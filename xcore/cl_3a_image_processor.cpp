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
#include "cl_macc_handler.h"
#include "cl_tnr_handler.h"
#include "cl_ee_handler.h"
#include "cl_dpc_handler.h"
#include "cl_bnr_handler.h"

#define XCAM_CL_3A_IMAGE_MAX_POOL_SIZE 6

namespace XCam {

CL3aImageProcessor::CL3aImageProcessor ()
    : CLImageProcessor ("CL3aImageProcessor")
    , _output_fourcc (V4L2_PIX_FMT_NV12)
    , _out_smaple_type (OutSampleYuv)
    , _hdr_mode (0)
    , _tnr_mode (0)
    , _enable_gamma (true)
    , _enable_macc (true)
    , _enable_dpc (false)
    , _snr_mode (XCAM_DENOISE_TYPE_SIMPLE | XCAM_DENOISE_TYPE_EE)
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
    if (result.ptr() == NULL)
        return false;

    switch (result->get_type ()) {
    case XCAM_3A_RESULT_WHITE_BALANCE:
    case XCAM_3A_RESULT_BLACK_LEVEL:
    case XCAM_3A_RESULT_R_GAMMA:
    case XCAM_3A_RESULT_G_GAMMA:
    case XCAM_3A_RESULT_B_GAMMA:
    case XCAM_3A_RESULT_RGB2YUV_MATRIX:
    case XCAM_3A_RESULT_DEFECT_PIXEL_CORRECTION:
    case XCAM_3A_RESULT_MACC:
    case XCAM_3A_RESULT_BAYER_NOISE_REDUCTION:
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

    if (result.ptr() == NULL)
        return XCAM_RETURN_BYPASS;

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
        _black_level->set_blc_config (bl_res->get_standard_result ());
        break;
    }

    case XCAM_3A_RESULT_DEFECT_PIXEL_CORRECTION: {
        SmartPtr<X3aDefectPixelResult> def_res = result.dynamic_cast_ptr<X3aDefectPixelResult> ();
        XCAM_ASSERT (def_res.ptr ());
        if (!_dpc.ptr())
            break;
        _dpc->set_dpc_config (def_res->get_standard_result ());
        break;
    }

    case XCAM_3A_RESULT_RGB2YUV_MATRIX: {
        SmartPtr<X3aColorMatrixResult> csc_res = result.dynamic_cast_ptr<X3aColorMatrixResult> ();
        XCAM_ASSERT (csc_res.ptr ());
        if (!_csc.ptr())
            break;
        _csc->set_rgbtoyuv_matrix (csc_res->get_standard_result ());
        break;
    }

    case XCAM_3A_RESULT_MACC: {
        SmartPtr<X3aMaccMatrixResult> macc_res = result.dynamic_cast_ptr<X3aMaccMatrixResult> ();
        XCAM_ASSERT (macc_res.ptr ());
        if (!_macc.ptr())
            break;
        _macc->set_macc_table (macc_res->get_standard_result ());
        break;
    }
    case XCAM_3A_RESULT_R_GAMMA:
    case XCAM_3A_RESULT_B_GAMMA:
        break;

    case XCAM_3A_RESULT_G_GAMMA:
    case XCAM_3A_RESULT_Y_GAMMA: {
        SmartPtr<X3aGammaTableResult> gamma_res = result.dynamic_cast_ptr<X3aGammaTableResult> ();
        XCAM_ASSERT (gamma_res.ptr ());
        if (!_gamma.ptr())
            break;
        _gamma->set_gamma_table (gamma_res->get_standard_result ());
        break;
    }

    case XCAM_3A_RESULT_TEMPORAL_NOISE_REDUCTION: {
        SmartPtr<X3aNoiseReductionResult> tnr_res = result.dynamic_cast_ptr<X3aNoiseReductionResult> ();
        XCAM_ASSERT (tnr_res.ptr ());
        if (!_tnr_yuv.ptr())
            break;
        float gain = ((XCam3aResultNoiseReduction)tnr_res->get_standard_result()).gain;
        float thr_y = ((XCam3aResultNoiseReduction)tnr_res->get_standard_result()).threshold1;
        float thr_uv = ((XCam3aResultNoiseReduction)tnr_res->get_standard_result()).threshold2;
        _tnr_yuv->set_gain (gain);
        _tnr_yuv->set_threshold (thr_y, thr_uv);
        break;
    }

    case XCAM_3A_RESULT_EDGE_ENHANCEMENT: {
        SmartPtr<X3aEdgeEnhancementResult> ee_ee_res = result.dynamic_cast_ptr<X3aEdgeEnhancementResult> ();
        XCAM_ASSERT (ee_ee_res.ptr ());
        if (!_ee.ptr())
            break;
        _ee->set_ee_config_ee (ee_ee_res->get_standard_result ());
        SmartPtr<X3aNoiseReductionResult> ee_nr_res = result.dynamic_cast_ptr<X3aNoiseReductionResult> ();
        XCAM_ASSERT (ee_nr_res.ptr ());
        if (!_ee.ptr())
            break;
        _ee->set_ee_config_nr (ee_nr_res->get_standard_result ());
        break;
    }

    case XCAM_3A_RESULT_BAYER_NOISE_REDUCTION: {
        SmartPtr<X3aBayerNoiseReduction> bnr_res = result.dynamic_cast_ptr<X3aBayerNoiseReduction> ();
        XCAM_ASSERT (bnr_res.ptr ());
        if (!_bnr.ptr())
            break;
        _bnr->set_bnr_config (bnr_res->get_standard_result ());
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
    _black_level = image_handler.dynamic_cast_ptr<CLBlcImageHandler> ();;
    XCAM_FAIL_RETURN (
        WARNING,
        image_handler.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CL3aImageProcessor create blc handler failed");
    add_handler (image_handler);

    image_handler = create_cl_dpc_image_handler (context);
    _dpc = image_handler.dynamic_cast_ptr<CLDpcImageHandler> ();;
    XCAM_FAIL_RETURN (
        WARNING,
        image_handler.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CL3aImageProcessor create dpc handler failed");
    _dpc->set_kernels_enable(_enable_dpc);
    add_handler (image_handler);

    image_handler = create_cl_bnr_image_handler (context);
    _bnr = image_handler.dynamic_cast_ptr<CLBnrImageHandler> ();;
    XCAM_FAIL_RETURN (
        WARNING,
        _bnr.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CL3aImageProcessor create bnr handler failed");
    _bnr->set_kernels_enable (XCAM_DENOISE_TYPE_BNR & _snr_mode);
    add_handler (image_handler);

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
    image_handler = create_cl_gamma_image_handler (context);
    _gamma = image_handler.dynamic_cast_ptr<CLGammaImageHandler> ();
    XCAM_FAIL_RETURN (
        WARNING,
        _gamma.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CL3aImageProcessor create gamma handler failed");
    _gamma->set_kernels_enable (_enable_gamma);
    add_handler (image_handler);

    /* hdr */
    image_handler = create_cl_hdr_image_handler (context, CL_HDR_DISABLE);
    _hdr = image_handler.dynamic_cast_ptr<CLHdrImageHandler> ();
    XCAM_FAIL_RETURN (
        WARNING,
        _hdr.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CL3aImageProcessor create hdr handler failed");
    if(_hdr_mode == CL_HDR_TYPE_RGB)
        _hdr->set_mode (_hdr_mode);
    add_handler (image_handler);

    /* demosaic */
    image_handler = create_cl_demosaic_image_handler (context);
    _demosaic = image_handler.dynamic_cast_ptr<CLBayer2RGBImageHandler> ();
    XCAM_FAIL_RETURN (
        WARNING,
        _demosaic.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CL3aImageProcessor create demosaic handler failed");
    image_handler->set_pool_size (XCAM_CL_3A_IMAGE_MAX_POOL_SIZE);
    add_handler (image_handler);

    /* hdr-lab*/
    image_handler = create_cl_hdr_image_handler (context, CL_HDR_DISABLE);
    _hdr = image_handler.dynamic_cast_ptr<CLHdrImageHandler> ();
    XCAM_FAIL_RETURN (
        WARNING,
        _hdr.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CL3aImageProcessor create hdr handler failed");
    if(_hdr_mode == CL_HDR_TYPE_LAB)
        _hdr->set_mode (_hdr_mode);
    add_handler (image_handler);

    /* bilateral noise reduction */
    image_handler = create_cl_denoise_image_handler (context);
    _binr = image_handler.dynamic_cast_ptr<CLDenoiseImageHandler> ();
    XCAM_FAIL_RETURN (
        WARNING,
        _binr.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CL3aImageProcessor create denoise handler failed");
    _binr->set_kernels_enable (XCAM_DENOISE_TYPE_BILATERAL & _snr_mode);
    image_handler->set_pool_size (XCAM_CL_3A_IMAGE_MAX_POOL_SIZE);
    add_handler (image_handler);

    /* Temporal Noise Reduction (RGB domain) */
    image_handler = create_cl_tnr_image_handler(context, CL_TNR_TYPE_RGB);
    _tnr_rgb = image_handler.dynamic_cast_ptr<CLTnrImageHandler> ();
    XCAM_FAIL_RETURN (
        WARNING,
        _tnr_rgb.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CL3aImageProcessor create tnr handler failed");
    _tnr_rgb->set_mode (CL_TNR_TYPE_RGB & _tnr_mode);
    add_handler (image_handler);

    /* simple noise reduction */
    image_handler = create_cl_snr_image_handler (context);
    _snr = image_handler.dynamic_cast_ptr<CLSnrImageHandler> ();
    XCAM_FAIL_RETURN (
        WARNING,
        _snr.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CL3aImageProcessor create snr handler failed");
    _snr->set_kernels_enable (XCAM_DENOISE_TYPE_SIMPLE & _snr_mode);
    add_handler (image_handler);

    /* macc */
    image_handler = create_cl_macc_image_handler (context);
    _macc = image_handler.dynamic_cast_ptr<CLMaccImageHandler> ();
    XCAM_FAIL_RETURN (
        WARNING,
        _macc.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CL3aImageProcessor create macc handler failed");
    _macc->set_kernels_enable (_enable_macc);
    add_handler (image_handler);

    /* color space conversion */
    image_handler = create_cl_csc_image_handler (context, CL_CSC_TYPE_RGBATONV12);
    _csc = image_handler.dynamic_cast_ptr<CLCscImageHandler> ();
    XCAM_FAIL_RETURN (
        WARNING,
        _csc .ptr (),
        XCAM_RETURN_ERROR_CL,
        "CL3aImageProcessor create csc handler failed");
    image_handler->set_pool_type (CLImageHandler::DrmBoPoolType);
    add_handler (image_handler);

    /* Temporal Noise Reduction (YUV domain) */
    image_handler = create_cl_tnr_image_handler(context, CL_TNR_TYPE_YUV);
    _tnr_yuv = image_handler.dynamic_cast_ptr<CLTnrImageHandler> ();
    XCAM_FAIL_RETURN (
        WARNING,
        _tnr_yuv.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CL3aImageProcessor create tnr handler failed");
    _tnr_yuv->set_mode (CL_TNR_TYPE_YUV & _tnr_mode);
    image_handler->set_pool_type (CLImageHandler::DrmBoPoolType);
    add_handler (image_handler);

    /* ee */
    image_handler = create_cl_ee_image_handler (context);
    _ee = image_handler.dynamic_cast_ptr<CLEeImageHandler> ();
    XCAM_FAIL_RETURN (
        WARNING,
        _ee.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CL3aImageProcessor create ee handler failed");
    _ee->set_kernels_enable (XCAM_DENOISE_TYPE_EE & _snr_mode);
    image_handler->set_pool_type (CLImageHandler::DrmBoPoolType);
    image_handler->set_pool_size (XCAM_CL_3A_IMAGE_MAX_POOL_SIZE);
    add_handler (image_handler);

    if (_out_smaple_type == OutSampleRGB) {
        image_handler = create_cl_csc_image_handler (context, CL_CSC_TYPE_NV12TORGBA);
        _csc = image_handler.dynamic_cast_ptr<CLCscImageHandler> ();
        XCAM_FAIL_RETURN (
            WARNING,
            _csc .ptr (),
            XCAM_RETURN_ERROR_CL,
            "CL3aImageProcessor create csc handler failed");
        image_handler->set_pool_type (CLImageHandler::DrmBoPoolType);
        image_handler->set_pool_size (XCAM_CL_3A_IMAGE_MAX_POOL_SIZE);
        add_handler (image_handler);
    }

    return XCAM_RETURN_NO_ERROR;
}

bool
CL3aImageProcessor::set_hdr (uint32_t mode)
{
    _hdr_mode = mode;

    STREAM_LOCK;

    if (_hdr.ptr ())
        return _hdr->set_mode (mode);

    return true;
}

bool
CL3aImageProcessor::set_gamma (bool enable)
{
    _enable_gamma = enable;

    STREAM_LOCK;

    if (_gamma.ptr ())
        return _gamma->set_kernels_enable (enable);

    return true;
}

bool
CL3aImageProcessor::set_denoise (uint32_t mode)
{
    _snr_mode = mode;

    STREAM_LOCK;

    if (_snr.ptr ())
        _snr->set_kernels_enable (XCAM_DENOISE_TYPE_SIMPLE & _snr_mode);
    if (_binr.ptr ())
        _binr->set_kernels_enable (XCAM_DENOISE_TYPE_BILATERAL & _snr_mode);
    if (_ee.ptr ())
        _ee->set_kernels_enable (XCAM_DENOISE_TYPE_EE & _snr_mode);
    if (_bnr.ptr ())
        _bnr->set_kernels_enable (XCAM_DENOISE_TYPE_BNR & _snr_mode);

    return true;
}

bool
CL3aImageProcessor::set_macc (bool enable)
{
    _enable_macc = enable;

    STREAM_LOCK;

    if (_macc.ptr ())
        return _macc->set_kernels_enable (enable);
    return true;
}

bool
CL3aImageProcessor::set_dpc (bool enable)
{
    _enable_dpc = enable;

    STREAM_LOCK;

    if (_dpc.ptr ())
        return _dpc->set_kernels_enable (enable);

    return true;
}

bool
CL3aImageProcessor::set_tnr (uint32_t mode, uint8_t level)
{
    _tnr_mode = mode;

    STREAM_LOCK;
    //TODO: map denoise level to threshold & gain
    XCAM_UNUSED(level);
    bool ret = false;
    if (_tnr_rgb.ptr ())
        ret = _tnr_rgb->set_kernels_enable (mode & CL_TNR_TYPE_RGB);
    if (_tnr_yuv.ptr ())
        ret = _tnr_yuv->set_kernels_enable (mode & CL_TNR_TYPE_YUV);

    return ret;
}
};
