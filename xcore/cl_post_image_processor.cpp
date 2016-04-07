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
#include "cl_csc_handler.h"

#define XCAM_CL_POST_IMAGE_DEFAULT_POOL_SIZE 6
#define XCAM_CL_POST_IMAGE_MAX_POOL_SIZE 12

namespace XCam {

CLPostImageProcessor::CLPostImageProcessor ()
    : CLImageProcessor ("CLPostImageProcessor")
    , _output_fourcc (V4L2_PIX_FMT_NV12)
    , _out_sample_type (OutSampleYuv)
    , _tnr_mode (TnrYuv)
    , _enable_retinex (false)
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

bool
CLPostImageProcessor::can_process_result (SmartPtr < X3aResult > & result)
{
    if (!result.ptr ())
        return false;

    switch (result->get_type ()) {
    case XCAM_3A_RESULT_TEMPORAL_NOISE_REDUCTION_YUV:
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
            if (_enable_retinex) {
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
    default:
        XCAM_LOG_WARNING ("CLPostImageProcessor unknow 3a result: %d", res_type);
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

    /* retinex */
    image_handler = create_cl_retinex_image_handler (context);
    _retinex = image_handler.dynamic_cast_ptr<CLRetinexImageHandler> ();
    XCAM_FAIL_RETURN (
        WARNING,
        _retinex.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CLPostImageProcessor create retinex handler failed");
    _retinex->set_kernels_enable (_enable_retinex);
    image_handler->set_pool_type (CLImageHandler::DrmBoPoolType);
    image_handler->set_pool_size (XCAM_CL_POST_IMAGE_MAX_POOL_SIZE);
    add_handler (image_handler);

    /* Temporal Noise Reduction */
    if (_enable_retinex) {
        switch (_tnr_mode) {
        case TnrYuv: {
            image_handler = create_cl_tnr_image_handler (context, CL_TNR_TYPE_YUV);
            _tnr = image_handler.dynamic_cast_ptr<CLTnrImageHandler> ();
            XCAM_FAIL_RETURN (
                WARNING,
                _tnr.ptr (),
                XCAM_RETURN_ERROR_CL,
                "CLPostImageProcessor create tnr handler failed");
            _tnr->set_mode (CL_TNR_TYPE_YUV);
            image_handler->set_pool_type (CLImageHandler::DrmBoPoolType);
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

    /* csc (nv12torgba) */
    image_handler = create_cl_csc_image_handler (context, CL_CSC_TYPE_NV12TORGBA);
    _csc = image_handler.dynamic_cast_ptr<CLCscImageHandler> ();
    XCAM_FAIL_RETURN (
        WARNING,
        _csc .ptr (),
        XCAM_RETURN_ERROR_CL,
        "CLPostImageProcessor create csc handler failed");
    _csc->set_kernels_enable (_out_sample_type == OutSampleRGB);
    _csc->set_output_format (_output_fourcc);
    image_handler->set_pool_type (CLImageHandler::DrmBoPoolType);
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
CLPostImageProcessor::set_retinex (bool enable)
{
    _enable_retinex = enable;

    STREAM_LOCK;

    return true;
}

};
