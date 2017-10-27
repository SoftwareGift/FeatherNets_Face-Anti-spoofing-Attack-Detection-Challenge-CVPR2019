/*
 * cl_post_image_processor.cpp - CL post image processor
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
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#include "cl_post_image_processor.h"
#include "cl_context.h"

#include "cl_tnr_handler.h"
#include "cl_retinex_handler.h"
#include "cl_defog_dcp_handler.h"
#include "cl_wavelet_denoise_handler.h"
#include "cl_newwavelet_denoise_handler.h"
#include "cl_3d_denoise_handler.h"
#include "cl_image_scaler.h"
#include "cl_wire_frame_handler.h"
#include "cl_csc_handler.h"
#include "cl_image_warp_handler.h"
#include "cl_image_360_stitch.h"
#include "cl_video_stabilizer.h"

#define XCAM_CL_POST_IMAGE_DEFAULT_POOL_SIZE 6
#define XCAM_CL_POST_IMAGE_MAX_POOL_SIZE 12

namespace XCam {

CLPostImageProcessor::CLPostImageProcessor ()
    : CLImageProcessor ("CLPostImageProcessor")
    , _output_fourcc (V4L2_PIX_FMT_NV12)
    , _out_sample_type (OutSampleYuv)
    , _scaler_factor (1.0)
    , _tnr_mode (TnrYuv)
    , _defog_mode (CLPostImageProcessor::DefogDisabled)
    , _wavelet_basis (CL_WAVELET_DISABLED)
    , _wavelet_channel (CL_IMAGE_CHANNEL_UV)
    , _wavelet_bayes_shrink (false)
    , _3d_denoise_mode (CLPostImageProcessor::Denoise3DDisabled)
    , _3d_denoise_ref_count (3)
    , _enable_scaler (false)
    , _enable_wireframe (false)
    , _enable_image_warp (false)
    , _enable_stitch (false)
    , _stitch_enable_seam (false)
    , _stitch_fisheye_map (false)
    , _stitch_lsc (false)
    , _stitch_fm_ocl (false)
    , _stitch_scale_mode (CLBlenderScaleLocal)
    , _stitch_width (0)
    , _stitch_height (0)
    , _stitch_res_mode (0)
    , _surround_mode (SphereView)
{
    XCAM_LOG_DEBUG ("CLPostImageProcessor constructed");
}

CLPostImageProcessor::~CLPostImageProcessor ()
{
    XCAM_LOG_DEBUG ("CLPostImageProcessor destructed");
}

bool
CLPostImageProcessor::set_output_format (uint32_t fourcc)
{
    switch (fourcc) {
    case XCAM_PIX_FMT_RGBA64:
    case V4L2_PIX_FMT_XBGR32:
    case V4L2_PIX_FMT_ABGR32:
    case V4L2_PIX_FMT_BGR32:
    case V4L2_PIX_FMT_RGB32:
    case V4L2_PIX_FMT_ARGB32:
    case V4L2_PIX_FMT_XRGB32:
        _out_sample_type = OutSampleRGB;
        break;
    case V4L2_PIX_FMT_NV12:
        _out_sample_type = OutSampleYuv;
        break;
    default:
        XCAM_LOG_WARNING (
            "cl post processor doesn't support output format: %s",
            xcam_fourcc_to_string(fourcc));
        return false;
    }

    _output_fourcc = fourcc;
    return true;
}

void
CLPostImageProcessor::set_stats_callback (const SmartPtr<StatsCallback> &callback)
{
    XCAM_ASSERT (callback.ptr ());
    _stats_callback = callback;
}

bool
CLPostImageProcessor::set_scaler_factor (const double factor)
{
    _scaler_factor = factor;

    return true;
}

bool
CLPostImageProcessor::can_process_result (SmartPtr < X3aResult > & result)
{
    if (!result.ptr ())
        return false;

    switch (result->get_type ()) {
    case XCAM_3A_RESULT_TEMPORAL_NOISE_REDUCTION_YUV:
    case XCAM_3A_RESULT_3D_NOISE_REDUCTION:
    case XCAM_3A_RESULT_WAVELET_NOISE_REDUCTION:
    case XCAM_3A_RESULT_FACE_DETECTION:
    case XCAM_3A_RESULT_DVS:
        return true;
    default:
        return false;
    }

    return false;
}

XCamReturn
CLPostImageProcessor::apply_3a_results (X3aResultList &results)
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
CLPostImageProcessor::apply_3a_result (SmartPtr<X3aResult> &result)
{
    STREAM_LOCK;

    if (!result.ptr ())
        return XCAM_RETURN_BYPASS;

    uint32_t res_type = result->get_type ();

    switch (res_type) {
    case XCAM_3A_RESULT_TEMPORAL_NOISE_REDUCTION_YUV: {
        SmartPtr<X3aTemporalNoiseReduction> tnr_res = result.dynamic_cast_ptr<X3aTemporalNoiseReduction> ();
        XCAM_ASSERT (tnr_res.ptr ());
        if (_tnr.ptr ()) {
            if (_defog_mode != CLPostImageProcessor::DefogDisabled) {
                XCam3aResultTemporalNoiseReduction config;
                xcam_mem_clear (config);
                // isp processor
                // config.gain = 0.12;

                // cl processor
                config.gain = 0.22;

                config.threshold [0] = 0.00081;
                config.threshold [1] = 0.00072;
                _tnr->set_yuv_config (config);
            } else {
                _tnr->set_yuv_config (tnr_res->get_standard_result ());
            }
        }
        break;
    }
    case XCAM_3A_RESULT_3D_NOISE_REDUCTION: {
        SmartPtr<X3aTemporalNoiseReduction> nr_res = result.dynamic_cast_ptr<X3aTemporalNoiseReduction> ();
        XCAM_ASSERT (nr_res.ptr ());
        if (_3d_denoise.ptr ()) {
            _3d_denoise->set_denoise_config (nr_res->get_standard_result ());
        }
        break;
    }
    case XCAM_3A_RESULT_WAVELET_NOISE_REDUCTION: {
        SmartPtr<X3aWaveletNoiseReduction> wavelet_res = result.dynamic_cast_ptr<X3aWaveletNoiseReduction> ();
        XCAM_ASSERT (wavelet_res.ptr ());
        if (_wavelet.ptr()) {
            _wavelet->set_denoise_config (wavelet_res->get_standard_result ());
        }
        if (_newwavelet.ptr()) {
            _newwavelet->set_denoise_config (wavelet_res->get_standard_result ());
        }
        break;
    }
    case XCAM_3A_RESULT_FACE_DETECTION: {
        SmartPtr<X3aFaceDetectionResult> fd_res = result.dynamic_cast_ptr<X3aFaceDetectionResult> ();
        XCAM_ASSERT (fd_res.ptr ());
        if (_wireframe.ptr ()) {
            _wireframe->set_wire_frame_config (fd_res->get_standard_result_ptr (), get_scaler_factor ());
        }
        break;
    }
    case XCAM_3A_RESULT_DVS: {
        SmartPtr<X3aDVSResult> dvs_res = result.dynamic_cast_ptr<X3aDVSResult> ();
        XCAM_ASSERT (dvs_res.ptr ());
        if (_image_warp.ptr ()) {
            _image_warp->set_warp_config (dvs_res->get_standard_result ());
        }
        break;
    }
    default:
        XCAM_LOG_WARNING ("CLPostImageProcessor unknown 3a result: %d", res_type);
        break;
    }

    return XCAM_RETURN_NO_ERROR;
}


XCamReturn
CLPostImageProcessor::create_handlers ()
{
    SmartPtr<CLImageHandler> image_handler;
    SmartPtr<CLContext> context = get_cl_context ();

    XCAM_ASSERT (context.ptr ());

    /* defog: retinex */
    image_handler = create_cl_retinex_image_handler (context);
    _retinex = image_handler.dynamic_cast_ptr<CLRetinexImageHandler> ();
    XCAM_FAIL_RETURN (
        WARNING,
        _retinex.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CLPostImageProcessor create retinex handler failed");
    _retinex->enable_handler (_defog_mode == CLPostImageProcessor::DefogRetinex);
    image_handler->set_pool_type (CLImageHandler::CLVideoPoolType);
    image_handler->set_pool_size (XCAM_CL_POST_IMAGE_MAX_POOL_SIZE);
    add_handler (image_handler);

    /* defog: dark channel prior */
    image_handler = create_cl_defog_dcp_image_handler (context);
    _defog_dcp = image_handler.dynamic_cast_ptr<CLDefogDcpImageHandler> ();
    XCAM_FAIL_RETURN (
        WARNING,
        _defog_dcp.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CLPostImageProcessor create defog handler failed");
    _defog_dcp->enable_handler (_defog_mode == CLPostImageProcessor::DefogDarkChannelPrior);
    image_handler->set_pool_type (CLImageHandler::CLVideoPoolType);
    image_handler->set_pool_size (XCAM_CL_POST_IMAGE_MAX_POOL_SIZE);
    add_handler (image_handler);

    /* Temporal Noise Reduction */
    if (_defog_mode != CLPostImageProcessor::DefogDisabled) {
        switch (_tnr_mode) {
        case TnrYuv: {
            image_handler = create_cl_tnr_image_handler (context, CL_TNR_TYPE_YUV);
            _tnr = image_handler.dynamic_cast_ptr<CLTnrImageHandler> ();
            XCAM_FAIL_RETURN (
                WARNING,
                _tnr.ptr (),
                XCAM_RETURN_ERROR_CL,
                "CLPostImageProcessor create tnr handler failed");
            image_handler->set_pool_type (CLImageHandler::CLVideoPoolType);
            image_handler->set_pool_size (XCAM_CL_POST_IMAGE_DEFAULT_POOL_SIZE);
            add_handler (image_handler);
            break;
        }
        case TnrDisable:
            XCAM_LOG_DEBUG ("CLPostImageProcessor disable tnr");
            break;
        default:
            XCAM_LOG_WARNING ("CLPostImageProcessor unknown tnr mode (%d)", _tnr_mode);
            break;
        }
    }

    /* wavelet denoise */
    switch (_wavelet_basis) {
    case CL_WAVELET_HAT: {
        image_handler = create_cl_wavelet_denoise_image_handler (context, _wavelet_channel);
        _wavelet = image_handler.dynamic_cast_ptr<CLWaveletDenoiseImageHandler> ();
        XCAM_FAIL_RETURN (
            WARNING,
            _wavelet.ptr (),
            XCAM_RETURN_ERROR_CL,
            "CLPostImageProcessor create wavelet denoise handler failed");
        _wavelet->enable_handler (true);
        image_handler->set_pool_type (CLImageHandler::CLVideoPoolType);
        image_handler->set_pool_size (XCAM_CL_POST_IMAGE_DEFAULT_POOL_SIZE);
        add_handler (image_handler);
        break;
    }
    case CL_WAVELET_HAAR: {
        image_handler = create_cl_newwavelet_denoise_image_handler (context, _wavelet_channel, _wavelet_bayes_shrink);
        _newwavelet = image_handler.dynamic_cast_ptr<CLNewWaveletDenoiseImageHandler> ();
        XCAM_FAIL_RETURN (
            WARNING,
            _newwavelet.ptr (),
            XCAM_RETURN_ERROR_CL,
            "CLPostImageProcessor create new wavelet denoise handler failed");
        _newwavelet->enable_handler (true);
        image_handler->set_pool_type (CLImageHandler::CLVideoPoolType);
        image_handler->set_pool_size (XCAM_CL_POST_IMAGE_DEFAULT_POOL_SIZE);
        add_handler (image_handler);
        break;
    }
    case CL_WAVELET_DISABLED:
    default :
        XCAM_LOG_DEBUG ("unknown or disable wavelet (%d)", _wavelet_basis);
        break;
    }

    /* 3D noise reduction */
    if (_3d_denoise_mode != CLPostImageProcessor::Denoise3DDisabled) {
        uint32_t denoise_channel = CL_IMAGE_CHANNEL_UV;

        if (_3d_denoise_mode == CLPostImageProcessor::Denoise3DUV) {
            denoise_channel = CL_IMAGE_CHANNEL_UV;
        } else if (_3d_denoise_mode == CLPostImageProcessor::Denoise3DYuv) {
            denoise_channel = CL_IMAGE_CHANNEL_Y | CL_IMAGE_CHANNEL_UV;
        }

        image_handler = create_cl_3d_denoise_image_handler (context, denoise_channel, _3d_denoise_ref_count);
        _3d_denoise = image_handler.dynamic_cast_ptr<CL3DDenoiseImageHandler> ();
        XCAM_FAIL_RETURN (
            WARNING,
            _3d_denoise.ptr (),
            XCAM_RETURN_ERROR_CL,
            "CL3aImageProcessor create 3D noise reduction handler failed");
        image_handler->set_pool_type (CLImageHandler::CLVideoPoolType);
        image_handler->set_pool_size (XCAM_CL_POST_IMAGE_MAX_POOL_SIZE);
        image_handler->enable_handler (true);
        add_handler (image_handler);
    }

    /* image scaler */
    image_handler = create_cl_image_scaler_handler (context, V4L2_PIX_FMT_NV12);
    _scaler = image_handler.dynamic_cast_ptr<CLImageScaler> ();
    XCAM_FAIL_RETURN (
        WARNING,
        _scaler.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CLPostImageProcessor create scaler handler failed");
    _scaler->set_scaler_factor (_scaler_factor, _scaler_factor);
    _scaler->set_buffer_callback (_stats_callback);
    image_handler->set_pool_type (CLImageHandler::CLVideoPoolType);
    image_handler->enable_handler (_enable_scaler);
    add_handler (image_handler);

    /* wire frame */
    image_handler = create_cl_wire_frame_image_handler (context);
    _wireframe = image_handler.dynamic_cast_ptr<CLWireFrameImageHandler> ();
    XCAM_FAIL_RETURN (
        WARNING,
        _wireframe.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CLPostImageProcessor create wire frame handler failed");
    _wireframe->enable_handler (_enable_wireframe);
    image_handler->set_pool_type (CLImageHandler::CLVideoPoolType);
    image_handler->set_pool_size (XCAM_CL_POST_IMAGE_DEFAULT_POOL_SIZE);
    add_handler (image_handler);

    /* image warp */
    image_handler = create_cl_image_warp_handler (context);
    _image_warp = image_handler.dynamic_cast_ptr<CLImageWarpHandler> ();
    XCAM_FAIL_RETURN (
        WARNING,
        _image_warp.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CLPostImageProcessor create image warp handler failed");
    _image_warp->enable_handler (_enable_image_warp);
    image_handler->set_pool_type (CLImageHandler::CLVideoPoolType);
    image_handler->set_pool_size (XCAM_CL_POST_IMAGE_MAX_POOL_SIZE);
    add_handler (image_handler);

    /* video stabilization */
    image_handler = create_cl_video_stab_handler (context);
    _video_stab = image_handler.dynamic_cast_ptr<CLVideoStabilizer> ();
    XCAM_FAIL_RETURN (
        WARNING,
        _video_stab.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CLPostImageProcessor create video stabilizer failed");
    _video_stab->enable_handler (false);
    image_handler->set_pool_type (CLImageHandler::CLVideoPoolType);
    image_handler->set_pool_size (XCAM_CL_POST_IMAGE_MAX_POOL_SIZE);
    add_handler (image_handler);

    /* image stitch */
    image_handler =
        create_image_360_stitch (context, _stitch_enable_seam, _stitch_scale_mode,
                                 _stitch_fisheye_map, _stitch_lsc, (SurroundMode) _surround_mode, (StitchResMode) _stitch_res_mode);
    _stitch = image_handler.dynamic_cast_ptr<CLImage360Stitch> ();
    XCAM_FAIL_RETURN (
        WARNING,
        _stitch.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CLPostImageProcessor create image stitch handler failed");
    _stitch->set_output_size (_stitch_width, _stitch_height);
#if HAVE_OPENCV
    _stitch->set_feature_match_ocl (_stitch_fm_ocl);
#endif
    image_handler->set_pool_type (CLImageHandler::CLVideoPoolType);
    image_handler->set_pool_size (XCAM_CL_POST_IMAGE_MAX_POOL_SIZE);
    image_handler->enable_handler (_enable_stitch);
    add_handler (image_handler);

    /* csc (nv12torgba) */
    image_handler = create_cl_csc_image_handler (context, CL_CSC_TYPE_NV12TORGBA);
    _csc = image_handler.dynamic_cast_ptr<CLCscImageHandler> ();
    XCAM_FAIL_RETURN (
        WARNING,
        _csc .ptr (),
        XCAM_RETURN_ERROR_CL,
        "CLPostImageProcessor create csc handler failed");
    _csc->enable_handler (_out_sample_type == OutSampleRGB);
    _csc->set_output_format (_output_fourcc);
    image_handler->set_pool_type (CLImageHandler::CLVideoPoolType);
    image_handler->set_pool_size (XCAM_CL_POST_IMAGE_DEFAULT_POOL_SIZE);
    add_handler (image_handler);

    return XCAM_RETURN_NO_ERROR;
}

bool
CLPostImageProcessor::set_tnr (CLTnrMode mode)
{
    _tnr_mode = mode;

    STREAM_LOCK;

    return true;
}

bool
CLPostImageProcessor::set_defog_mode (CLDefogMode mode)
{
    _defog_mode = mode;

    STREAM_LOCK;

    return true;
}

bool
CLPostImageProcessor::set_wavelet (CLWaveletBasis basis, uint32_t channel, bool bayes_shrink)
{
    _wavelet_basis = basis;
    _wavelet_channel = (CLImageChannel) channel;
    _wavelet_bayes_shrink = bayes_shrink;

    STREAM_LOCK;

    return true;
}

bool
CLPostImageProcessor::set_3ddenoise_mode (CL3DDenoiseMode mode, uint8_t ref_frame_count)
{
    _3d_denoise_mode = mode;
    _3d_denoise_ref_count = ref_frame_count;

    STREAM_LOCK;

    return true;
}

bool
CLPostImageProcessor::set_scaler (bool enable)
{
    _enable_scaler = enable;

    STREAM_LOCK;

    return true;
}

bool
CLPostImageProcessor::set_wireframe (bool enable)
{
    _enable_wireframe = enable;

    STREAM_LOCK;

    return true;
}

bool
CLPostImageProcessor::set_image_warp (bool enable)
{
    _enable_image_warp = enable;

    STREAM_LOCK;

    return true;
}

bool
CLPostImageProcessor::set_image_stitch (
    bool enable_stitch, bool enable_seam, CLBlenderScaleMode scale_mode, bool enable_fisheye_map,
    bool lsc, bool fm_ocl, uint32_t stitch_width, uint32_t stitch_height, uint32_t res_mode)
{
    XCAM_ASSERT (scale_mode < CLBlenderScaleMax);

    _enable_stitch = enable_stitch;
    if (enable_stitch)
        _stitch_enable_seam = enable_seam;
    else
        _stitch_enable_seam = false;

    _stitch_scale_mode = scale_mode;
    _stitch_fisheye_map = enable_fisheye_map;
    _stitch_lsc = lsc;
    _stitch_width = stitch_width;
    _stitch_height = stitch_height;
    _stitch_res_mode = res_mode;

#if HAVE_OPENCV
    _stitch_fm_ocl = fm_ocl;
#else
    XCAM_UNUSED (fm_ocl);
#endif

    STREAM_LOCK;

    return true;
}

};
