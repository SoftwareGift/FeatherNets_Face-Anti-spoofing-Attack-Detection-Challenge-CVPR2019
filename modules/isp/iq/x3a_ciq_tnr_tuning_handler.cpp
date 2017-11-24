/*
 * x3a_ciq_tnr_tuning_handler.cpp - x3a Common IQ TNR tuning handler
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
 * Author: Zong Wei <wei.zong@intel.com>
 */

#include "x3a_analyzer.h"
#include "x3a_ciq_tuning_handler.h"
#include "x3a_ciq_tnr_tuning_handler.h"

namespace XCam {

typedef struct _X3aCiqTnrTuningStaticData {
    double analog_gain;
    double yuv_gain;
    double y_threshold;
    double uv_threshold;
    double rgb_gain;
    double r_threshold;
    double g_threshold;
    double b_threshold;
} X3aCiqTnrTuningStaticData;

const X3aCiqTnrTuningStaticData imx185_tuning[X3A_CIQ_GAIN_STEPS] = {
    {1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
    {16.98, 0.8, 0.0081, 0.00725, 0.2, 0.0253, 0.0158, 0.0168},
    {49.55, 0.5, 0.0146, 0.0128, 0.4, 0.0434, 0.0274, 0.0317},
    {139.63, 0.3, 0.0247, 0.0253, 0.8, 0.0602, 0.0377, 0.0445},
    {X3A_CIQ_GAIN_MAX, 0.2, 0.0358, 0.0329, 1.0, 0.0994, 0.0696, 0.0924},
};

X3aCiqTnrTuningHandler::X3aCiqTnrTuningHandler ()
    : X3aCiqTuningHandler ("X3aCiqTnrTuningHandler")
{
}

X3aCiqTnrTuningHandler::~X3aCiqTnrTuningHandler ()
{
}

XCamReturn
X3aCiqTnrTuningHandler::analyze (X3aResultList &output)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    const X3aCiqTnrTuningStaticData* tuning = imx185_tuning;
    if (NULL != _tuning_data) {
        tuning = (X3aCiqTnrTuningStaticData*)_tuning_data;;
    }

    XCam3aResultTemporalNoiseReduction config;
    SmartPtr<X3aTemporalNoiseReduction> nr_result = new X3aTemporalNoiseReduction (XCAM_3A_RESULT_3D_NOISE_REDUCTION);
    SmartPtr<X3aTemporalNoiseReduction> yuv_result = new X3aTemporalNoiseReduction (XCAM_3A_RESULT_TEMPORAL_NOISE_REDUCTION_YUV);

    int64_t et = get_current_exposure_time ();
    double analog_gain = get_current_analog_gain ();
    double max_analog_gain = get_max_analog_gain ();
    XCAM_UNUSED (et);
    XCAM_UNUSED (max_analog_gain);
    XCAM_LOG_DEBUG ("get current AG = (%f), max AG = (%f), et = (%" PRId64 ")", analog_gain, max_analog_gain, et);

    uint8_t i_curr = 0;
    uint8_t i_prev = 0;
    for (i_curr = 0; i_curr < X3A_CIQ_GAIN_STEPS; i_curr++) {
        if (analog_gain <= tuning[i_curr].analog_gain) {
            break;
        }
        i_prev = i_curr;
    }
    if (i_curr >= X3A_CIQ_GAIN_STEPS) {
        i_curr = X3A_CIQ_GAIN_STEPS - 1;
    }

    //Calculate YUV config
    xcam_mem_clear (config);
    config.gain = linear_interpolate_p2 (tuning[i_prev].yuv_gain, tuning[i_curr].yuv_gain,
                                         tuning[i_prev].analog_gain, tuning[i_curr].analog_gain, analog_gain);

    config.threshold[0] = linear_interpolate_p2 (tuning[i_prev].y_threshold, tuning[i_curr].y_threshold,
                          tuning[i_prev].analog_gain, tuning[i_curr].analog_gain, analog_gain);

    config.threshold[1] = linear_interpolate_p2 (tuning[i_prev].uv_threshold, tuning[i_curr].uv_threshold,
                          tuning[i_prev].analog_gain, tuning[i_curr].analog_gain, analog_gain);

    config.threshold[2] = 0.0;
    XCAM_LOG_DEBUG ("Calculate YUV temporal noise reduction config: yuv_gain(%f), y_threshold(%f), uv_threshold(%f)",
                    config.gain, config.threshold[0], config.threshold[1]);

    yuv_result->set_standard_result (config);
    output.push_back (yuv_result);

    //Calculate 3D NR config
    xcam_mem_clear (config);
    config.gain = linear_interpolate_p2 (tuning[i_prev].rgb_gain, tuning[i_curr].rgb_gain,
                                         tuning[i_prev].analog_gain, tuning[i_curr].analog_gain, analog_gain);

    config.threshold[0] = linear_interpolate_p2 (tuning[i_prev].r_threshold, tuning[i_curr].r_threshold,
                          tuning[i_prev].analog_gain, tuning[i_curr].analog_gain, analog_gain);

    config.threshold[1] = linear_interpolate_p2 (tuning[i_prev].g_threshold, tuning[i_curr].g_threshold,
                          tuning[i_prev].analog_gain, tuning[i_curr].analog_gain, analog_gain);

    config.threshold[2] = linear_interpolate_p2 (tuning[i_prev].b_threshold, tuning[i_curr].b_threshold,
                          tuning[i_prev].analog_gain, tuning[i_curr].analog_gain, analog_gain);

    XCAM_LOG_DEBUG ("Calculate 3D noise reduction config: gain(%f), y_threshold(%f), uv_threshold(%f)",
                    config.gain, config.threshold[0], config.threshold[1]);

    nr_result->set_standard_result (config);
    output.push_back (nr_result);

    return ret;
}

};
