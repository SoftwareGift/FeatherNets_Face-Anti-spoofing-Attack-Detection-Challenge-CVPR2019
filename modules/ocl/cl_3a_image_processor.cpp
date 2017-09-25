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
#include "cl_csc_handler.h"
#include "cl_bayer_pipe_handler.h"
#include "cl_yuv_pipe_handler.h"
#if ENABLE_YEENR_HANDLER
#include "cl_ee_handler.h"
#endif
#include "cl_tnr_handler.h"
#include "cl_tonemapping_handler.h"
#include "cl_newtonemapping_handler.h"
#include "cl_bayer_basic_handler.h"

#define XCAM_CL_3A_IMAGE_MAX_POOL_SIZE 6

namespace XCam {

CL3aImageProcessor::CL3aImageProcessor ()
    : CLImageProcessor ("CL3aImageProcessor")
    , _output_fourcc (V4L2_PIX_FMT_NV12)
    , _3a_stats_bits (8)
    , _pipeline_profile (BasicPipelineProfile)
    , _capture_stage (TonemappingStage)
    , _wdr_mode (WDRdisabled)
    , _tnr_mode (0)
    , _enable_gamma (true)
    , _enable_macc (true)
    , _snr_mode (0)
{
    keep_attached_buf (true);
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
    XCAM_FAIL_RETURN (
        WARNING,
        V4L2_PIX_FMT_NV12 == fourcc,
        false,
        "cl 3a processor doesn't support output format: %s",
        xcam_fourcc_to_string (fourcc));

    _output_fourcc = fourcc;
    return true;
}

bool
CL3aImageProcessor::set_capture_stage (CaptureStage capture_stage)
{
    _capture_stage = capture_stage;
    return true;
}

bool
CL3aImageProcessor::set_3a_stats_bits (uint32_t bits)
{
    switch (bits) {
    case 8:
    case 12:
        _3a_stats_bits = bits;
        break;
    default:
        XCAM_LOG_WARNING ("cl image processor 3a stats doesn't support %d-bits", bits);
        return false;
    }
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
    case XCAM_3A_RESULT_BRIGHTNESS:
    case XCAM_3A_RESULT_3D_NOISE_REDUCTION:
    case XCAM_3A_RESULT_TEMPORAL_NOISE_REDUCTION_YUV:
    case XCAM_3A_RESULT_EDGE_ENHANCEMENT:
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
        if (_bayer_basic_pipe.ptr ()) {
            _bayer_basic_pipe->set_wb_config (wb_res->get_standard_result ());
            _bayer_basic_pipe->set_3a_result (result);
        }
        break;
    }

    case XCAM_3A_RESULT_BLACK_LEVEL: {
        SmartPtr<X3aBlackLevelResult> bl_res = result.dynamic_cast_ptr<X3aBlackLevelResult> ();
        XCAM_ASSERT (bl_res.ptr ());
        if (_bayer_basic_pipe.ptr ()) {
            _bayer_basic_pipe->set_blc_config (bl_res->get_standard_result ());
            _bayer_basic_pipe->set_3a_result (result);
        }
        break;
    }

    case XCAM_3A_RESULT_DEFECT_PIXEL_CORRECTION: {
        SmartPtr<X3aDefectPixelResult> def_res = result.dynamic_cast_ptr<X3aDefectPixelResult> ();
        XCAM_ASSERT (def_res.ptr ());
        XCAM_UNUSED (def_res);
        break;
    }

    case XCAM_3A_RESULT_RGB2YUV_MATRIX: {
        SmartPtr<X3aColorMatrixResult> csc_res = result.dynamic_cast_ptr<X3aColorMatrixResult> ();
        XCAM_ASSERT (csc_res.ptr ());
        if (_csc.ptr()) {
            _csc->set_matrix (csc_res->get_standard_result ());
            _csc->set_3a_result (result);
        }
        if (_yuv_pipe.ptr()) {
            _yuv_pipe->set_rgbtoyuv_matrix (csc_res->get_standard_result ());
            _yuv_pipe->set_3a_result (result);
        }
        break;
    }

    case XCAM_3A_RESULT_MACC: {
        SmartPtr<X3aMaccMatrixResult> macc_res = result.dynamic_cast_ptr<X3aMaccMatrixResult> ();
        XCAM_ASSERT (macc_res.ptr ());
        if (_yuv_pipe.ptr()) {
            _yuv_pipe->set_macc_table (macc_res->get_standard_result ());
            _yuv_pipe->set_3a_result (result);
        }
        break;
    }
    case XCAM_3A_RESULT_R_GAMMA:
    case XCAM_3A_RESULT_B_GAMMA:
        break;

    case XCAM_3A_RESULT_G_GAMMA:
    case XCAM_3A_RESULT_Y_GAMMA: {
        SmartPtr<X3aGammaTableResult> gamma_res = result.dynamic_cast_ptr<X3aGammaTableResult> ();
        XCAM_ASSERT (gamma_res.ptr ());
        if (_bayer_basic_pipe.ptr ()) {
            _bayer_basic_pipe->set_gamma_table (gamma_res->get_standard_result ());
            _bayer_basic_pipe->set_3a_result (result);
        }
        break;
    }

    case XCAM_3A_RESULT_3D_NOISE_REDUCTION: {
        SmartPtr<X3aTemporalNoiseReduction> nr_res = result.dynamic_cast_ptr<X3aTemporalNoiseReduction> ();
        XCAM_ASSERT (nr_res.ptr ());
        XCAM_UNUSED (nr_res);

        break;
    }

    case XCAM_3A_RESULT_TEMPORAL_NOISE_REDUCTION_YUV: {
        SmartPtr<X3aTemporalNoiseReduction> tnr_res = result.dynamic_cast_ptr<X3aTemporalNoiseReduction> ();
        XCAM_ASSERT (tnr_res.ptr ());
        if (_yuv_pipe.ptr ()) {
            _yuv_pipe->set_tnr_yuv_config(tnr_res->get_standard_result ());
            _yuv_pipe->set_3a_result (result);
        }
        break;
    }

    case XCAM_3A_RESULT_EDGE_ENHANCEMENT: {
        SmartPtr<X3aEdgeEnhancementResult> ee_ee_res = result.dynamic_cast_ptr<X3aEdgeEnhancementResult> ();
        XCAM_ASSERT (ee_ee_res.ptr ());
        if (_bayer_pipe.ptr()) {
            _bayer_pipe->set_ee_config (ee_ee_res->get_standard_result ());
            _bayer_pipe->set_3a_result (result);
        }
#if ENABLE_YEENR_HANDLER
        if (_ee.ptr()) {
            _ee->set_ee_config_ee (ee_ee_res->get_standard_result ());
            _ee->set_3a_result (result);
        }
#endif
        break;
    }

    case XCAM_3A_RESULT_BAYER_NOISE_REDUCTION: {
        SmartPtr<X3aBayerNoiseReduction> bnr_res = result.dynamic_cast_ptr<X3aBayerNoiseReduction> ();
        XCAM_ASSERT (bnr_res.ptr ());
        if (_bayer_pipe.ptr()) {
            _bayer_pipe->set_bnr_config (bnr_res->get_standard_result ());
            _bayer_pipe->set_3a_result (result);
        }

        break;
    }

    case XCAM_3A_RESULT_BRIGHTNESS: {
        SmartPtr<X3aBrightnessResult> brightness_res = result.dynamic_cast_ptr<X3aBrightnessResult> ();
        XCAM_ASSERT (brightness_res.ptr ());
        float brightness_level = ((XCam3aResultBrightness)brightness_res->get_standard_result()).brightness_level;
        XCAM_UNUSED (brightness_level);
        break;
    }

    default:
        XCAM_LOG_WARNING ("CL3aImageProcessor unknown 3a result:%d", res_type);
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

    /* bayer pipeline */
    image_handler = create_cl_bayer_basic_image_handler (context, _enable_gamma, _3a_stats_bits);
    _bayer_basic_pipe = image_handler.dynamic_cast_ptr<CLBayerBasicImageHandler> ();
    XCAM_FAIL_RETURN (
        WARNING,
        _bayer_basic_pipe.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CL3aImageProcessor create bayer basic pipe handler failed");
    image_handler->set_pool_size (XCAM_CL_3A_IMAGE_MAX_POOL_SIZE);
    _bayer_basic_pipe->set_stats_callback (_stats_callback);
    add_handler (image_handler);

    /* tone mapping */
    switch(_wdr_mode) {
    case Gaussian: {
        image_handler = create_cl_tonemapping_image_handler (context);
        _tonemapping = image_handler.dynamic_cast_ptr<CLTonemappingImageHandler> ();
        XCAM_FAIL_RETURN (
            WARNING,
            _tonemapping.ptr (),
            XCAM_RETURN_ERROR_CL,
            "CL3aImageProcessor create tonemapping handler failed");
        _tonemapping->enable_handler (true);
        image_handler->set_pool_size (XCAM_CL_3A_IMAGE_MAX_POOL_SIZE);
        add_handler (image_handler);
        break;
    }
    case Haleq: {
        image_handler = create_cl_newtonemapping_image_handler (context);
        _newtonemapping = image_handler.dynamic_cast_ptr<CLNewTonemappingImageHandler> ();
        XCAM_FAIL_RETURN (
            WARNING,
            _newtonemapping.ptr (),
            XCAM_RETURN_ERROR_CL,
            "CL3aImageProcessor create tonemapping handler failed");
        _newtonemapping->enable_handler (true);
        image_handler->set_pool_size (XCAM_CL_3A_IMAGE_MAX_POOL_SIZE);
        add_handler (image_handler);
        break;
    }
    default:
        XCAM_LOG_DEBUG ("WDR disabled");
        break;
    }

    /* bayer pipe */
    image_handler = create_cl_bayer_pipe_image_handler (context);
    _bayer_pipe = image_handler.dynamic_cast_ptr<CLBayerPipeImageHandler> ();
    XCAM_FAIL_RETURN (
        WARNING,
        image_handler.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CL3aImageProcessor create bayer pipe handler failed");

    _bayer_pipe->enable_denoise (XCAM_DENOISE_TYPE_BNR & _snr_mode);
    image_handler->set_pool_size (XCAM_CL_3A_IMAGE_MAX_POOL_SIZE * 2);
    add_handler (image_handler);
    if(_capture_stage == BasicbayerStage)
        return XCAM_RETURN_NO_ERROR;

    image_handler = create_cl_yuv_pipe_image_handler (context);
    _yuv_pipe = image_handler.dynamic_cast_ptr<CLYuvPipeImageHandler> ();
    XCAM_FAIL_RETURN (
        WARNING,
        _yuv_pipe.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CL3aImageProcessor create yuv pipe handler failed");
    _yuv_pipe->set_tnr_enable (_tnr_mode & CL_TNR_TYPE_YUV);
    image_handler->set_pool_size (XCAM_CL_3A_IMAGE_MAX_POOL_SIZE * 2);
    add_handler (image_handler);

#if ENABLE_YEENR_HANDLER
    /* ee */
    image_handler = create_cl_ee_image_handler (context);
    _ee = image_handler.dynamic_cast_ptr<CLEeImageHandler> ();
    XCAM_FAIL_RETURN (
        WARNING,
        _ee.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CL3aImageProcessor create ee handler failed");
    _ee->enable_handler (XCAM_DENOISE_TYPE_EE & _snr_mode);
    image_handler->set_pool_type (CLImageHandler::CLVideoPoolType);
    image_handler->set_pool_size (XCAM_CL_3A_IMAGE_MAX_POOL_SIZE);
    add_handler (image_handler);
#endif

    XCAM_FAIL_RETURN (
        WARNING,
        post_config (),
        XCAM_RETURN_ERROR_CL,
        "CL3aImageProcessor post_config failed");

    return XCAM_RETURN_NO_ERROR;
}

bool
CL3aImageProcessor::post_config ()
{
    CLImageProcessor::ImageHandlerList::iterator i_handler = handlers_begin ();
    CLImageProcessor::ImageHandlerList::iterator end = handlers_end ();
    uint32_t swap_y_count = 0;
    bool start_count = false;
    bool ret = true;

    if (!_yuv_pipe.ptr ())  //not necessary to check
        return true;

    for (; i_handler != end; ++i_handler) {
        if (!start_count) {
            SmartPtr<CLYuvPipeImageHandler> convert_yuv = (*i_handler).dynamic_cast_ptr<CLYuvPipeImageHandler> ();
            if (convert_yuv.ptr () && convert_yuv.ptr () == _yuv_pipe.ptr ())
                start_count = true;
            continue;
        }

        SmartPtr<CLCloneImageHandler> clone_y = (*i_handler).dynamic_cast_ptr<CLCloneImageHandler> ();
        if (clone_y.ptr () && clone_y->is_handler_enabled () && (clone_y->get_clone_flags () & SwappedBuffer::SwapY))
            swap_y_count++;
    }

    if (swap_y_count % 2 == 1)
        ret = _yuv_pipe->enable_buf_pool_swap_flags (SwappedBuffer::SwapY | SwappedBuffer::SwapUV, SwappedBuffer::OrderY1Y0 | SwappedBuffer::OrderUV0UV1);
    else
        ret = _yuv_pipe->enable_buf_pool_swap_flags (SwappedBuffer::SwapY | SwappedBuffer::SwapUV, SwappedBuffer::OrderY0Y1 | SwappedBuffer::OrderUV0UV1);

    return ret;
}

bool
CL3aImageProcessor::set_profile (const CL3aImageProcessor::PipelineProfile value)
{
    _pipeline_profile = value;

    if (value >= AdvancedPipelineProfile)
        _tnr_mode |= CL_TNR_TYPE_YUV;

    if (value >= ExtremePipelineProfile) {
        _snr_mode |= XCAM_DENOISE_TYPE_BNR;
    }
    STREAM_LOCK;
    if (_yuv_pipe.ptr ())
        _yuv_pipe->set_tnr_enable (_tnr_mode & CL_TNR_TYPE_YUV);

    return true;
}

bool
CL3aImageProcessor::set_gamma (bool enable)
{
    _enable_gamma = enable;

    STREAM_LOCK;

    return true;
}

bool
CL3aImageProcessor::set_denoise (uint32_t mode)
{
    _snr_mode = mode;

    STREAM_LOCK;
    if (_bayer_pipe.ptr ())
        _bayer_pipe->enable_denoise (XCAM_DENOISE_TYPE_BNR & _snr_mode);

    return true;
}

bool
CL3aImageProcessor::set_macc (bool enable)
{
    _enable_macc = enable;

    STREAM_LOCK;
    return true;
}

bool
CL3aImageProcessor::set_tonemapping (CLTonemappingMode wdr_mode)
{
    _wdr_mode = wdr_mode;

    STREAM_LOCK;

    return true;
}

bool
CL3aImageProcessor::set_tnr (uint32_t mode, uint8_t level)
{
    XCAM_UNUSED (level);
    _tnr_mode = mode;

    STREAM_LOCK;
    if (_yuv_pipe.ptr ())
        _yuv_pipe->set_tnr_enable (_tnr_mode & CL_TNR_TYPE_YUV);

    return true;
}

};
