/*
 * xcam_params.h - 3A parameters
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

#ifndef C_XCAM_PARAMS_H
#define C_XCAM_PARAMS_H

#include <base/xcam_defs.h>
#include <base/xcam_common.h>
#include <base/xcam_3a_types.h>

XCAM_BEGIN_DECLARE

typedef struct _XCamAeParam {
    XCamAeMode              mode;
    XCamAeMeteringMode      metering_mode;
    XCam3AWindow            window;
    XCam3AWindow            window_list[XCAM_AE_MAX_METERING_WINDOW_COUNT];
    XCamFlickerMode         flicker_mode;
    /* speed, default 1.0 */
    double                  speed;

    /* exposure limitation */
    uint64_t                exposure_time_min, exposure_time_max;
    double                  max_analog_gain;

    /* exposure manual values */
    uint64_t                manual_exposure_time;
    double                  manual_analog_gain;

    double                  aperture_fn;

    /*ev*/
    double                  ev_shift;
} XCamAeParam;

typedef struct _XCamAwbParam {
    XCamAwbMode             mode;
    /* speed, default 1.0 */
    double                  speed;
    uint32_t                cct_min, cct_max;
    XCam3AWindow            window;

    /* manual gain, default 0.0 */
    double                  gr_gain;
    double                  r_gain;
    double                  b_gain;
    double                  gb_gain;
} XCamAwbParam;

typedef struct _XCamAfParam {

} XCamAfParam;

typedef struct _XCamCommonParam {
    /* R, G, B gamma table, size = XCAM_GAMMA_TABLE_SIZE */
    bool                      is_manual_gamma;
    double                    r_gamma [XCAM_GAMMA_TABLE_SIZE];
    double                    g_gamma [XCAM_GAMMA_TABLE_SIZE];
    double                    b_gamma [XCAM_GAMMA_TABLE_SIZE];

    /*
     * manual brightness, contrast, hue, saturation, sharpness
     * -1.0 < value < 1.0
     */
    double                     nr_level;
    double                     tnr_level;

    double                     brightness;
    double                     contrast;
    double                     hue;
    double                     saturation;
    double                     sharpness;

    /* others */
    bool                       enable_dvs;
    bool                       enable_gbce;
    bool                       enable_night_mode;
    XCamColorEffect            color_effect;
} XCamCommonParam;

typedef struct _XCamSmartAnalysisParam {
    uint32_t   width;
    uint32_t   height;
    double     fps;

} XCamSmartAnalysisParam;

XCAM_END_DECLARE

#endif //C_XCAM_PARAMS_H
