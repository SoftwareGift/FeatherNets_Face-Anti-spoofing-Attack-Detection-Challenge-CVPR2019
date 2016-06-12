/*
 * isp_image_processor.cpp - isp image processor
 *
 *  Copyright (c) 2014-2015 Intel Corporation
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

#include "isp_image_processor.h"
#include "x3a_isp_config.h"
#include "isp_controller.h"
#include "isp_config_translator.h"

namespace XCam {

IspImageProcessor::IspImageProcessor (SmartPtr<IspController> &controller)
    : ImageProcessor ("IspImageProcessor")
    , _controller (controller)
    , _3a_config (new X3aIspConfig)
{
    _sensor = new SensorDescriptor;
    _translator = new IspConfigTranslator (_sensor);
    XCAM_LOG_DEBUG ("IspImageProcessor construction");
}

IspImageProcessor::~IspImageProcessor ()
{
    XCAM_LOG_DEBUG ("~IspImageProcessor destruction");
}

XCamReturn
IspImageProcessor::process_buffer(SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output)
{
    output = input;
    return XCAM_RETURN_NO_ERROR;
}

bool
IspImageProcessor::can_process_result (SmartPtr<X3aResult> &result)
{
    if (result.ptr() == NULL)
        return false;

    switch (result->get_type()) {
    case X3aIspConfig::IspExposureParameters:
    case X3aIspConfig::IspAllParameters:
    case XCAM_3A_RESULT_WHITE_BALANCE:
    case XCAM_3A_RESULT_EXPOSURE:
    case XCAM_3A_RESULT_BLACK_LEVEL:
    case XCAM_3A_RESULT_YUV2RGB_MATRIX:
    case XCAM_3A_RESULT_RGB2YUV_MATRIX:
    case XCAM_3A_RESULT_R_GAMMA:
    case XCAM_3A_RESULT_G_GAMMA:
    case XCAM_3A_RESULT_B_GAMMA:
    case XCAM_3A_RESULT_MACC:
    case XCAM_3A_RESULT_BAYER_NOISE_REDUCTION:
        return true;
    default:
        return false;
    }

    return false;
}

XCamReturn
IspImageProcessor::apply_3a_results (X3aResultList &results)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    if (results.empty())
        return XCAM_RETURN_ERROR_PARAM;

    // activate sensor to make translator work
    if (!_sensor->is_ready()) {
        struct atomisp_sensor_mode_data sensor_data;
        xcam_mem_clear (sensor_data);
        if (_controller->get_sensor_mode_data(sensor_data) != XCAM_RETURN_NO_ERROR) {
            XCAM_LOG_WARNING ("ispimageprocessor initiliaze sensor failed");
        } else
            _sensor->set_sensor_data (sensor_data);
        XCAM_ASSERT (_sensor->is_ready());
    }

    if ((ret = merge_results (results)) != XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_WARNING ("merge 3a result to isp config failed");
        return XCAM_RETURN_ERROR_UNKNOWN;
    }

    if ((ret = apply_exposure_result (results)) != XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_WARNING ("set 3a exposure to sensor failed");
    }

    // check _3a_config
    XCAM_ASSERT (_3a_config.ptr());
    XCAM_ASSERT (_controller.ptr());
    ret = _controller->set_3a_config (_3a_config.ptr());
    if (ret != XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_WARNING ("set 3a config to isp failed");
    }
    _3a_config->clear ();
    return ret;
}

XCamReturn
IspImageProcessor::apply_3a_result (SmartPtr<X3aResult> &result)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    X3aResultList results;
    results.push_back (result);
    ret = apply_3a_results (results);
    return ret;
}

XCamReturn
IspImageProcessor::merge_results (X3aResultList &results)
{
    if (results.empty())
        return XCAM_RETURN_ERROR_PARAM;

    for (X3aResultList::iterator iter = results.begin ();
            iter != results.end ();)
    {
        SmartPtr<X3aResult> &x3a_result = *iter;
        if (_3a_config->attach (x3a_result, _translator.ptr())) {
            x3a_result->set_done (true);
            results.erase (iter++);
        } else
            ++iter;
    }
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
IspImageProcessor::apply_exposure_result (X3aResultList &results)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    for (X3aResultList::iterator iter = results.begin ();
            iter != results.end ();)
    {
        if ((*iter)->get_type() == X3aIspConfig::IspExposureParameters) {
            SmartPtr<X3aIspExposureResult> res = (*iter).dynamic_cast_ptr<X3aIspExposureResult> ();
            if (!res.ptr () ||
                    ((ret = _controller->set_3a_exposure (res.ptr ())) != XCAM_RETURN_NO_ERROR)) {
                XCAM_LOG_WARNING ("set 3a exposure to sensor failed");
            }
            if (res.ptr ())
                res->set_done (true);
            results.erase (iter++);
        } else if ((*iter)->get_type() == XCAM_3A_RESULT_EXPOSURE) {
            SmartPtr<X3aExposureResult> res = (*iter).dynamic_cast_ptr<X3aExposureResult> ();
            struct atomisp_exposure isp_exposure;
            xcam_mem_clear (isp_exposure);
            XCAM_ASSERT (res.ptr ());
            ret = _translator->translate_exposure (res->get_standard_result (), isp_exposure);
            if (ret != XCAM_RETURN_NO_ERROR) {
                XCAM_LOG_WARNING ("translate 3a exposure to sensor failed");
            }
            if ((ret = _controller->set_3a_exposure (isp_exposure)) != XCAM_RETURN_NO_ERROR) {
                XCAM_LOG_WARNING ("set 3a exposure to sensor failed");
            }
            res->set_done (true);
            results.erase (iter++);
        } else
            ++iter;
    }
    return XCAM_RETURN_NO_ERROR;
}

IspExposureImageProcessor::IspExposureImageProcessor (SmartPtr<IspController> &controller)
    : IspImageProcessor (controller)
{
}

bool
IspExposureImageProcessor::can_process_result (SmartPtr<X3aResult> &result)
{
    if (result.ptr() == NULL)
        return false;

    switch (result->get_type()) {
    case XCAM_3A_RESULT_EXPOSURE:
        return true;

    default:
        return false;
    }

    return false;
}

};
