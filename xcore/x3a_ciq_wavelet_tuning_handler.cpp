/*
 * x3a_ciq_wavelet_tuning_handler.cpp - x3a Common IQ Wavelet denoise tuning handler
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
#include "x3a_ciq_wavelet_tuning_handler.h"

namespace XCam {

typedef struct _X3aCiqWaveletTuningStaticData {
    double analog_gain;
    double hard_threshold;
    double soft_threshold;
    uint8_t  decom_levels;
} X3aCiqWaveletTuningStaticData;

const X3aCiqWaveletTuningStaticData imx185_tuning[X3A_CIQ_GAIN_STEPS] = {
    {1.0, 0.01, 1.0, 5},
    {16.98, 0.02, 0.7, 5},
    {49.55, 0.2, 0.5, 5},
    {139.63, 0.3, 0.3, 5},
    {X3A_CIQ_GAIN_MAX, 0.5, 0.2, 5},
};

X3aCiqWaveletTuningHandler::X3aCiqWaveletTuningHandler ()
    : X3aCiqTuningHandler ("X3aCiqWaveletTuningHandler")
{
}

X3aCiqWaveletTuningHandler::~X3aCiqWaveletTuningHandler ()
{
}

XCamReturn
X3aCiqWaveletTuningHandler::analyze (X3aResultList &output)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    const X3aCiqWaveletTuningStaticData* tuning = imx185_tuning;
    if (NULL != _tuning_data) {
        tuning = (X3aCiqWaveletTuningStaticData*)_tuning_data;;
    }

    XCam3aResultWaveletNoiseReduction config;
    SmartPtr<X3aWaveletNoiseReduction> settings = new X3aWaveletNoiseReduction (XCAM_3A_RESULT_WAVELET_NOISE_REDUCTION);

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

    //Calculate Wavelet denoise config
    xcam_mem_clear (config);

    /* [0]:soft threshold / [1]:hard threshold */
    config.threshold[0] = linear_interpolate_p2 (tuning[i_prev].soft_threshold, tuning[i_curr].soft_threshold,
                          tuning[i_prev].analog_gain, tuning[i_curr].analog_gain, analog_gain);

    config.threshold[1] = linear_interpolate_p2 (tuning[i_prev].hard_threshold, tuning[i_curr].hard_threshold,
                          tuning[i_prev].analog_gain, tuning[i_curr].analog_gain, analog_gain);

    config.decomposition_levels = 1;

    config.analog_gain = analog_gain / X3A_CIQ_GAIN_MAX;
    XCAM_LOG_DEBUG ("Calculate Wavelet noise reduction config: soft threshold(%f), hard threshold(%f), decomposition levels(%d)",
                    config.threshold[0], config.threshold[1], config.decomposition_levels);

    settings->set_standard_result (config);
    output.push_back (settings);

    return ret;
}

};
