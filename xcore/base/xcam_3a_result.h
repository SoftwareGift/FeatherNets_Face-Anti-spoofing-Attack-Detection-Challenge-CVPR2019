/*
 * xcam_3a_result.h - 3A result interface
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
 *         Zong Wei <wei.zong@intel.com>
 */

#ifndef C_XCAM_3A_RESULT_H
#define C_XCAM_3A_RESULT_H

#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <base/xcam_defs.h>

XCAM_BEGIN_DECLARE

#define XCAM_3A_MAX_RESULT_COUNT 256
#define xcam_3a_result_type(result)  (((XCam3aResultHead*)result)->type)

typedef enum _ImageProcessType {
    XCAM_IMAGE_PROCESS_ONCE,
    XCAM_IMAGE_PROCESS_ALWAYS,
    XCAM_IMAGE_PROCESS_POST,
} XCamImageProcessType;

typedef enum _XCam3aResultType {
    XCAM_3A_RESULT_NULL            = 0,
    /* White Balance */
    XCAM_3A_RESULT_WHITE_BALANCE,
    XCAM_3A_RESULT_BLACK_LEVEL,
    XCAM_3A_RESULT_YUV2RGB_MATRIX,
    XCAM_3A_RESULT_RGB2YUV_MATRIX,

    /* Exposure */
    XCAM_3A_RESULT_EXPOSURE,

    /* Focus */
    XCAM_3A_RESULT_FOCUS,

    XCAM_3A_RESULT_DEMOSAIC,
    //XCAM_3A_RESULT_EIGEN_COLOR_DEMOSAICING,
    XCAM_3A_RESULT_DEFECT_PIXEL_CORRECTION,

    /* noise reduction */
    XCAM_3A_RESULT_NOISE_REDUCTION,
    XCAM_3A_RESULT_3D_NOISE_REDUCTION,
    XCAM_3A_RESULT_TEMPORAL_NOISE_REDUCTION_YUV,
    XCAM_3A_RESULT_LUMA_NOISE_REDUCTION,
    XCAM_3A_RESULT_ADVANCED_NOISE_REDUCTION,
    XCAM_3A_RESULT_CHROMA_NOISER_EDUCTION,
    XCAM_3A_RESULT_BAYER_NOISE_REDUCTION,
    XCAM_3A_RESULT_WAVELET_NOISE_REDUCTION,

    XCAM_3A_RESULT_EDGE_ENHANCEMENT,
    //XCAM_3A_RESULT_FRIGLE_CONTROL,
    XCAM_3A_RESULT_MACC,
    //XCAM_3A_RESULT_MACCTABLE,
    XCAM_3A_RESULT_CHROMA_TONE_CONTROL,
    //XCAM_3A_RESULT_CHROMATONECONTROLTABLE,
    XCAM_3A_RESULT_CHROMA_ENHANCEMENT,
    XCAM_3A_RESULT_Y_GAMMA,
    XCAM_3A_RESULT_R_GAMMA,
    XCAM_3A_RESULT_G_GAMMA,
    XCAM_3A_RESULT_B_GAMMA,
    XCAM_3A_RESULT_BRIGHTNESS,
    //XCAM_3A_RESULT_SHADING_TABLE,

    //Smart Analysis Type
    XCAM_3A_RESULT_FACE_DETECTION = 0x4000,
    XCAM_3A_RESULT_DVS,

    XCAM_3A_RESULT_USER_DEFINED_TYPE = 0x8000,
} XCam3aResultType;

/* matrix size 3x3 */
#define XCAM_COLOR_MATRIX_SIZE 9
#define XCAM_GAMMA_TABLE_SIZE 256
#define XCAM_CHROMA_AXIS_SIZE 16
#define XCAM_CHROMA_MATRIX_SIZE 4
#define XCAM_BNR_TABLE_SIZE 64

typedef struct _XCam3aResultHead XCam3aResultHead;

struct _XCam3aResultHead {
    XCam3aResultType      type;
    XCamImageProcessType  process_type;
    uint32_t              version;
    void                  (*destroy) (XCam3aResultHead *);
};

typedef struct _XCam3aResultWhiteBalance {
    XCam3aResultHead head;

    /* data */
    double           r_gain;
    double           gr_gain;
    double           gb_gain;
    double           b_gain;
} XCam3aResultWhiteBalance;

typedef struct _XCam3aResultBlackLevel {
    XCam3aResultHead head;

    /* data */
    double           r_level;
    double           gr_level;
    double           gb_level;
    double           b_level;
} XCam3aResultBlackLevel;

typedef struct _XCam3aResultColorMatrix {
    XCam3aResultHead head;

    /* data */
    double           matrix [XCAM_COLOR_MATRIX_SIZE];
} XCam3aResultColorMatrix;

typedef struct _XCam3aResultExposure {
    XCam3aResultHead head;

    /* data */
    int32_t          exposure_time; //in micro seconds
    double           analog_gain;   // multipler
    double           digital_gain;  // multipler
    double           aperture;      //fn
} XCam3aResultExposure;

typedef struct _XCam3aResultFocus {
    XCam3aResultHead head;

    /* data */
    int32_t          position;
} XCam3aResultFocus;

typedef struct _XCam3aResultDemosaic {
    XCam3aResultHead head;

    /* data */
    double           noise;
    double           threshold_cr;
    double           threshold_cb;
} XCam3aResultDemosaic;


/* DefectPixel Correction */
typedef struct _XCam3aResultDefectPixel {
    XCam3aResultHead head;

    /* data */
    double           gain;
    double           gr_threshold;
    double           r_threshold;
    double           b_threshold;
    double           gb_threshold;
} XCam3aResultDefectPixel;

typedef struct _XCam3aResultNoiseReduction {
    XCam3aResultHead head;

    /* data */
    double           gain;
    double           threshold1;
    double           threshold2;
} XCam3aResultNoiseReduction;

typedef struct _XCam3aResultBayerNoiseReduction {
    XCam3aResultHead head;

    /* data */
    double           bnr_gain;
    double           direction;
    double           table[XCAM_BNR_TABLE_SIZE];
} XCam3aResultBayerNoiseReduction;

typedef struct _XCam3aResultEdgeEnhancement {
    XCam3aResultHead head;

    /* data */
    double           gain;
    double           threshold;
} XCam3aResultEdgeEnhancement;

typedef struct _XCam3aResultGammaTable {
    XCam3aResultHead head;

    /* data */
    double           table[XCAM_GAMMA_TABLE_SIZE];
} XCam3aResultGammaTable;

typedef struct _XCam3aResultMaccMatrix {
    XCam3aResultHead head;

    /* data */
    double           table[XCAM_CHROMA_AXIS_SIZE * XCAM_CHROMA_MATRIX_SIZE];
} XCam3aResultMaccMatrix;

typedef struct _XCam3aResultChromaToneControl {
    XCam3aResultHead head;

    /* data */
    double           uv_gain [XCAM_GAMMA_TABLE_SIZE]; // according to Y
} XCam3aResultChromaToneControl;

typedef struct _XCam3aResultBrightness {
    XCam3aResultHead head;

    /* data */
    double           brightness_level; // range [-1,1], -1 is full dark , 0 is normal val, 1 is full bright
} XCam3aResultBrightness;

typedef struct _XCam3aResultTemporalNoiseReduction {
    XCam3aResultHead head;

    /* data */
    double           gain;
    double           threshold[3];
} XCam3aResultTemporalNoiseReduction;

typedef struct _XCam3aResultWaveletNoiseReduction {
    XCam3aResultHead head;

    /* data */
    uint8_t          decomposition_levels;
    double           threshold[2];  /* [0]:soft threshold / [1]:hard threshold */
    double           analog_gain;
} XCam3aResultWaveletNoiseReduction;

XCAM_END_DECLARE

#endif
