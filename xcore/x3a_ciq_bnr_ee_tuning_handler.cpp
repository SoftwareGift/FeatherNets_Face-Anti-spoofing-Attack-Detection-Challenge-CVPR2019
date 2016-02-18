/*
 * x3a_ciq_bnr_ee_tuning_handler.cpp - x3a Common IQ Bayer NR EE tuning handler
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
 * Author: Wangfei <feix.w.wang@intel.com>
 */

#include "x3a_analyzer.h"
#include "x3a_ciq_tuning_handler.h"
#include "x3a_ciq_bnr_ee_tuning_handler.h"

namespace XCam {

typedef struct _X3aCiqBnrEeTuningStaticData {
    double analog_gain;
    double ee_gain;
    double ee_threshold;
} X3aCiqBnrEeTuningStaticData;

double table_2_0[XCAM_BNR_TABLE_SIZE] = {
    3.978874, 3.966789, 3.930753, 3.871418, 3.789852, 3.687501, 3.566151, 3.427876, 3.274977, 3.109920,
    2.935268, 2.753622, 2.567547, 2.379525, 2.191896, 2.006815, 1.826218, 1.651792, 1.484965, 1.326889,
    1.178449, 1.040267, 0.912718, 0.795950, 0.689911, 0.594371, 0.508957, 0.433173, 0.366437, 0.308103,
    0.257483, 0.213875, 0.176575, 0.144896, 0.118179, 0.095804, 0.077194, 0.061822, 0.049210, 0.038934,
    0.030617, 0.023930, 0.018591, 0.014355, 0.011017, 0.008404, 0.006372, 0.004802, 0.003597, 0.002678,
    0.001981, 0.001457, 0.001065, 0.000774, 0.000559, 0.000401, 0.000286, 0.000203, 0.000143, 0.000100,
    0.000070, 0.000048, 0.000033, 0.000023
};

double table_0_0_5[XCAM_BNR_TABLE_SIZE] = {
    63.661991, 60.628166, 52.366924, 41.023067, 29.146584, 18.781729, 10.976704, 6.000000, 6.000000, 6.000000,
    6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000,
    6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000,
    6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000,
    6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000,
    6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000, 6.000000,
    6.000000, 6.000000, 6.000000, 6.000000
};

const X3aCiqBnrEeTuningStaticData imx185_tuning[X3A_CIQ_EE_GAIN_STEPS] = {
    {1.0, 2.5, 0.008},
    {4.0, 1.8, 0.012},
    {16.98, 1.1, 0.02},
    {49.55, 0.8, 0.06},
    {139.63, 0.07, 0.1},
    {X3A_CIQ_GAIN_MAX, 0.03, 0.4},
};

X3aCiqBnrEeTuningHandler::X3aCiqBnrEeTuningHandler ()
    : X3aCiqTuningHandler ("X3aCiqBnrEeTuningHandler")
{
}

X3aCiqBnrEeTuningHandler::~X3aCiqBnrEeTuningHandler ()
{
}

XCamReturn
X3aCiqBnrEeTuningHandler::analyze (X3aResultList &output)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    const X3aCiqBnrEeTuningStaticData* tuning = imx185_tuning;
    if (NULL != _tuning_data) {
        tuning = (X3aCiqBnrEeTuningStaticData*)_tuning_data;;
    }

    XCam3aResultBayerNoiseReduction bnr_config;
    XCam3aResultEdgeEnhancement ee_config;
    SmartPtr<X3aBayerNoiseReduction> bnr_result = new X3aBayerNoiseReduction (XCAM_3A_RESULT_BAYER_NOISE_REDUCTION);
    SmartPtr<X3aEdgeEnhancementResult> ee_result = new X3aEdgeEnhancementResult (XCAM_3A_RESULT_EDGE_ENHANCEMENT);

    double analog_gain = get_current_analog_gain ();

    uint8_t i_curr = 0;
    uint8_t i_prev = 0;
    for (i_curr = 0; i_curr < X3A_CIQ_EE_GAIN_STEPS; i_curr++) {
        if (analog_gain <= tuning[i_curr].analog_gain) {
            break;
        }
        i_prev = i_curr;
    }
    if (i_curr >= X3A_CIQ_EE_GAIN_STEPS) {
        i_curr = X3A_CIQ_EE_GAIN_STEPS - 1;
    }

    xcam_mem_clear (bnr_config);
    xcam_mem_clear (ee_config);

    ee_config.gain = linear_interpolate_p2 (tuning[i_prev].ee_gain, tuning[i_curr].ee_gain,
                                            tuning[i_prev].analog_gain, tuning[i_curr].analog_gain, analog_gain);

    ee_config.threshold = linear_interpolate_p2 (tuning[i_prev].ee_threshold, tuning[i_curr].ee_threshold,
                          tuning[i_prev].analog_gain, tuning[i_curr].analog_gain, analog_gain);

    ee_result->set_standard_result (ee_config);
    output.push_back (ee_result);

    if(i_curr < 3)
        memcpy(bnr_config.table, table_0_0_5, XCAM_BNR_TABLE_SIZE * sizeof(double));
    else
        memcpy(bnr_config.table, table_2_0, XCAM_BNR_TABLE_SIZE * sizeof(double));

    bnr_result->set_standard_result (bnr_config);
    output.push_back (bnr_result);

    return ret;
}

};

