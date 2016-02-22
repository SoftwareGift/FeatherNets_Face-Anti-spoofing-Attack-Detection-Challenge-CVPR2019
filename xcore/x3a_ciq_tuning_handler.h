/*
 * x3a_ciq_tuning_handler.h - x3a Common IQ tuning handler
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

#ifndef XCAM_3A_CIQ_TUNING_HANDLER_H
#define XCAM_3A_CIQ_TUNING_HANDLER_H

#include "handler_interface.h"

namespace XCam {

#define X3A_CIQ_PIXEL_DEPTH 10

#define X3A_CIQ_EXPOSURE_TIME_STEPS  4 //Number of Exposure Time steps
#define X3A_CIQ_EXPOSURE_TIME_MAX    40000 //Max ET in microseconds (40ms)
#define X3A_CIQ_EXPOSURE_TIME_TICK   (X3A_CIQ_EXPOSURE_TIME_MAX / X3A_CIQ_EXPOSURE_TIME_STEPS)

#define X3A_CIQ_EE_GAIN_STEPS 6 //Number of EE Gain steps
#define X3A_CIQ_GAIN_STEPS 5 //Number of Gain steps
#define X3A_CIQ_GAIN_MAX   249 //Max Gain

#define X3A_CIQ_LSC_LUT_WIDTH    16
#define X3A_CIQ_LSC_LUT_HEIGHT   9
#define X3A_CIQ_LSC_LUT_SIZE    (16 * 9)

typedef enum _X3aCiqBayerOrder {
    X3A_CIQ_RGrGbB = 0,
    X3A_CIQ_GrRBGb = 1,
    X3A_CIQ_GbBRGr = 2,
    X3A_CIQ_BGbGrR = 3,
} X3aCiqBayerOrder;

typedef enum _X3aCiqCIEIlluminants {
    X3A_CIQ_ILLUMINANT_HALO = 0,  // Incandescent / Tungsten
    X3A_CIQ_ILLUMINANT_F2 =   1,    // Cool White Fluorescent
    X3A_CIQ_ILLUMINANT_F11 =  2,   // Philips TL84
    X3A_CIQ_ILLUMINANT_D50 =  3,   // Horizon Light
    X3A_CIQ_ILLUMINANT_D65 =  4,   // Noon Daylight
    X3A_CIQ_ILLUMINANT_D75 =  5,   // North sky Daylight
    X3A_CIQ_ILLUMINANT_COUNT
} X3aCiqCIEIlluminants;

typedef struct _X3aCiqCIEIlluminantsTable
{
    X3aCiqCIEIlluminants CIEIlluminantIndex;
    uint16_t CCT;
} X3aCiqCIEIlluminantsTable;

static const X3aCiqCIEIlluminantsTable X3a_Ciq_illuminants_table[X3A_CIQ_ILLUMINANT_COUNT] =
{
    {X3A_CIQ_ILLUMINANT_HALO, 2100},
    {X3A_CIQ_ILLUMINANT_F2, 3000},
    {X3A_CIQ_ILLUMINANT_F11, 4051},
    {X3A_CIQ_ILLUMINANT_D50, 5000},
    {X3A_CIQ_ILLUMINANT_D65, 6500},
    {X3A_CIQ_ILLUMINANT_D75, 7500},
};

class X3aCiqTuningHandler
    : public AnalyzerHandler
{
public:
    explicit X3aCiqTuningHandler (const char *name = NULL);
    virtual ~X3aCiqTuningHandler ();

    void set_tuning_data (void* data);
    void set_ae_handler (SmartPtr<AeHandler> &handler);
    void set_awb_handler (SmartPtr<AwbHandler> &handler);

    double get_max_analog_gain ();
    double get_current_analog_gain ();
    int64_t get_current_exposure_time ();
    uint32_t get_current_estimate_cct ();

private:
    XCAM_DEAD_COPY (X3aCiqTuningHandler);

protected:
    const void           *_tuning_data;

private:
    char                 *_name;
    SmartPtr<AeHandler>   _ae_handler;
    SmartPtr<AwbHandler>  _awb_handler;
};

};

#endif // XCAM_3A_CIQ_TUNING_HANDLER_H
