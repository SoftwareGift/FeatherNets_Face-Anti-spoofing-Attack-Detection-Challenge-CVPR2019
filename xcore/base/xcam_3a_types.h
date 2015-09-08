/*
 *  Copyright (c) 2014 Intel Corporation
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

/*!
 * \file xcam_3a_types.h
 * \brief 3A interface variable types
 */

#ifndef __XCAM_3A_TYPES_H
#define __XCAM_3A_TYPES_H

#include <string.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <base/xcam_defs.h>

XCAM_BEGIN_DECLARE

typedef enum  {
    XCAM_MODE_NONE = -1,
    XCAM_MODE_PREVIEW = 0,
    XCAM_MODE_CAPTURE = 1,
    XCAM_MODE_VIDEO = 2,
    XCAM_MODE_CONTINUOUS_CAPTURE = 3
} XCamMode;

typedef enum {
    XCAM_AE_MODE_NOT_SET = -1,
    XCAM_AE_MODE_AUTO,
    XCAM_AE_MODE_MANUAL,
    XCAM_AE_MODE_SHUTTER_PRIORITY,
    XCAM_AE_MODE_APERTURE_PRIORITY
} XCamAeMode;

#define XCAM_AE_MAX_METERING_WINDOW_COUNT 6

typedef enum {
    XCAM_AE_METERING_MODE_AUTO,   /*mode_evaluative*/
    XCAM_AE_METERING_MODE_SPOT,   /*window*/
    XCAM_AE_METERING_MODE_CENTER,  /*mode_center*/
    XCAM_AE_METERING_MODE_WEIGHTED_WINDOW, /* weighted_window */
} XCamAeMeteringMode;

typedef enum {
    XCAM_SCENE_MODE_NOT_SET = -1,
    XCAM_SCENE_MODE_AUTO,
    XCAM_SCENE_MODE_PORTRAIT,
    XCAM_SCENE_MODE_SPORTS,
    XCAM_SCENE_MODE_LANDSCAPE,
    XCAM_SCENE_MODE_NIGHT,
    XCAM_SCENE_MODE_NIGHT_PORTRAIT,
    XCAM_SCENE_MODE_FIREWORKS,
    XCAM_SCENE_MODE_TEXT,
    XCAM_SCENE_MODE_SUNSET,
    XCAM_SCENE_MODE_PARTY,
    XCAM_SCENE_MODE_CANDLELIGHT,
    XCAM_SCENE_MODE_BEACH_SNOW,
    XCAM_SCENE_MODE_DAWN_DUSK,
    XCAM_SCENE_MODE_FALL_COLORS,
    XCAM_SCENE_MODE_BACKLIGHT
} XCamSceneMode;

typedef enum {
    XCAM_AWB_MODE_NOT_SET = -1,
    XCAM_AWB_MODE_AUTO = 0,
    XCAM_AWB_MODE_MANUAL,
    XCAM_AWB_MODE_DAYLIGHT,
    XCAM_AWB_MODE_SUNSET,
    XCAM_AWB_MODE_CLOUDY,
    XCAM_AWB_MODE_TUNGSTEN,
    XCAM_AWB_MODE_FLUORESCENT,
    XCAM_AWB_MODE_WARM_FLUORESCENT,
    XCAM_AWB_MODE_SHADOW,
    XCAM_AWB_MODE_WARM_INCANDESCENT
} XCamAwbMode;

typedef enum {
    XCAM_AE_ISO_MODE_AUTO,   /* Automatic */
    XCAM_AE_ISO_MODE_MANUAL  /* Manual */
} XCamIsoMode;

typedef enum {
    XCAM_AE_FLICKER_MODE_AUTO,
    XCAM_AE_FLICKER_MODE_50HZ,
    XCAM_AE_FLICKER_MODE_60HZ,
    XCAM_AE_FLICKER_MODE_OFF
} XCamFlickerMode;

#if 0
typedef enum {
    XCAM_AF_MODE_NOT_SET = -1,
    XCAM_AF_MODE_AUTO,
    XCAM_AF_MODE_MACRO,
    XCAM_AF_MODE_INFINITY,
    XCAM_AF_MODE_FIXED,
    XCAM_AF_MODE_MANUAL,
    XCAM_AF_MODE_CONTINUOUS
} XCamAfMode;
#endif

/*! \brief XCam3AWindow.
 * Represents a rectangle area. Could be converted to
 * AIQ ia_rectangle, see convert_xcam_to_ia_window().
 */
typedef struct _XCam3AWindow {
    int32_t x_start; /*!< X of start point (left-upper corner) */
    int32_t y_start; /*!< Y of start point (left-upper corner) */
    int32_t x_end;   /*!< X of end point (right-bottom corner) */
    int32_t y_end;   /*!< Y of start point (left-upper corner) */
    int weight;
} XCam3AWindow;

typedef struct _XCamExposureResult {
    int64_t time_in_us;
    double analog_gain;
    double digital_gain;
    double aperture_fn;
    int32_t iso;
} XCamExposureResult;

typedef enum {
    XCAM_COLOR_EFFECT_NONE,
    XCAM_COLOR_EFFECT_SKY_BLUE,
    XCAM_COLOR_EFFECT_SKIN_WHITEN_LOW,
    XCAM_COLOR_EFFECT_SKIN_WHITEN,
    XCAM_COLOR_EFFECT_SKIN_WHITEN_HIGH,
    XCAM_COLOR_EFFECT_SEPIA,
    XCAM_COLOR_EFFECT_NEGATIVE,
    XCAM_COLOR_EFFECT_GRAYSCALE,
} XCamColorEffect;

typedef enum {
    XCAM_DENOISE_TYPE_SIMPLE    = (1UL << 0), // simple noise reduction
    XCAM_DENOISE_TYPE_BILATERAL = (1UL << 1), // bilateral noise reduction
    XCAM_DENOISE_TYPE_EE        = (1UL << 2), // luminance noise reduction and edge enhancement
    XCAM_DENOISE_TYPE_BNR       = (1UL << 3), // bayer noise reduction
    XCAM_DENOISE_TYPE_ANR       = (1UL << 4), // advanced bayer noise reduction
    XCAM_DENOISE_TYPE_BIYUV     = (1UL << 5), // bilateral on yuv noise reduction
} XCamDenoiseType;

XCAM_END_DECLARE

#endif //__XCAM_3A_TYPES_H

