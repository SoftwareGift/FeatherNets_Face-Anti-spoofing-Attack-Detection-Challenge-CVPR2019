/*
 * x3a_result_factory.cpp - 3A result factory
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

#include "x3a_result_factory.h"

namespace XCam {

#define XCAM_3A_RESULT_FACTORY(DataType, res_type, from)            \
    DataType *ret =                                                 \
        new DataType (res_type);                                    \
    if (from) {                                                     \
        uint32_t type = xcam_3a_result_type (from);                 \
        if (type != XCAM_3A_RESULT_NULL && type != res_type) {      \
            XCAM_ASSERT (false);                                    \
            XCAM_LOG_WARNING ("create result from wrong type:%d to type:%d", type, res_type);    \
        }                                                           \
        ret->set_standard_result (*from);                           \
    }                                                               \
    return ret;


Mutex X3aResultFactory::_mutex;
SmartPtr<X3aResultFactory> X3aResultFactory::_instance (NULL);

SmartPtr<X3aResultFactory>
X3aResultFactory::instance ()
{
    SmartLock locker (_mutex);
    if (_instance.ptr ())
        return _instance;

    _instance = new X3aResultFactory;
    return _instance;
}

X3aResultFactory::X3aResultFactory ()
{
}

X3aResultFactory::~X3aResultFactory ()
{
}

SmartPtr<X3aResult>
X3aResultFactory::create_3a_result (XCam3aResultHead *from)
{
    SmartPtr<X3aResult> result (NULL);

    XCAM_ASSERT (from);
    if (!from)
        return result;

    uint32_t type = xcam_3a_result_type (from);

    switch (type) {
    case XCAM_3A_RESULT_WHITE_BALANCE:
        result = create_whitebalance ((XCam3aResultWhiteBalance*)from);
        break;
    case XCAM_3A_RESULT_BLACK_LEVEL:
        result = create_blacklevel ((XCam3aResultBlackLevel*)from);
        break;
    case XCAM_3A_RESULT_YUV2RGB_MATRIX:
        result = create_yuv2rgb_colormatrix ((XCam3aResultColorMatrix*)from);
        break;
    case XCAM_3A_RESULT_RGB2YUV_MATRIX:
        result = create_rgb2yuv_colormatrix ((XCam3aResultColorMatrix*)from);
        break;
    case XCAM_3A_RESULT_EXPOSURE:
        result = create_exposure ((XCam3aResultExposure*)from);
        break;
    case XCAM_3A_RESULT_FOCUS:
        result = create_focus ((XCam3aResultFocus*)from);
        break;
    case XCAM_3A_RESULT_DEMOSAIC:
        result = create_demosaicing ((XCam3aResultDemosaic*)from);
        break;
    case XCAM_3A_RESULT_DEFECT_PIXEL_CORRECTION:
        result = create_defectpixel ((XCam3aResultDefectPixel*)from);
        break;
    case XCAM_3A_RESULT_NOISE_REDUCTION:
        result = create_noise_reduction ((XCam3aResultNoiseReduction*)from);
        break;
    case XCAM_3A_RESULT_3D_NOISE_REDUCTION:
        result = create_3d_noise_reduction ((XCam3aResultTemporalNoiseReduction*)from);
        break;
    case XCAM_3A_RESULT_TEMPORAL_NOISE_REDUCTION_YUV:
        result = create_yuv_temp_noise_reduction ((XCam3aResultTemporalNoiseReduction*)from);
        break;
    case XCAM_3A_RESULT_EDGE_ENHANCEMENT:
        result = create_edge_enhancement ((XCam3aResultEdgeEnhancement*)from);
        break;
    case XCAM_3A_RESULT_MACC:
        result = create_macc ((XCam3aResultMaccMatrix*)from);
        break;
    case XCAM_3A_RESULT_CHROMA_TONE_CONTROL:
        result = create_chroma_tone_control ((XCam3aResultChromaToneControl*)from);
        break;
    case XCAM_3A_RESULT_Y_GAMMA:
        result = create_y_gamma_table ((XCam3aResultGammaTable*)from);
        break;
    case XCAM_3A_RESULT_R_GAMMA:
        result = create_r_gamma_table ((XCam3aResultGammaTable*)from);
        break;
    case XCAM_3A_RESULT_G_GAMMA:
        result = create_g_gamma_table ((XCam3aResultGammaTable*)from);
        break;
    case XCAM_3A_RESULT_B_GAMMA:
        result = create_b_gamma_table ((XCam3aResultGammaTable*)from);
        break;
    case XCAM_3A_RESULT_BAYER_NOISE_REDUCTION:
        result = create_bayer_noise_reduction ((XCam3aResultBayerNoiseReduction*)from);
        break;
    case XCAM_3A_RESULT_BRIGHTNESS:
        result = create_brightness ((XCam3aResultBrightness*)from);
        break;
    case XCAM_3A_RESULT_WAVELET_NOISE_REDUCTION:
        result = create_wavelet_noise_reduction ((XCam3aResultWaveletNoiseReduction*)from);
        break;
    case XCAM_3A_RESULT_FACE_DETECTION:
        result = create_face_detection ((XCamFDResult*)from);
        break;
    case XCAM_3A_RESULT_DVS:
        result = create_digital_video_stabilizer ((XCamDVSResult*)from);
        break;
    default:
        XCAM_LOG_WARNING ("create 3a result with unknown result type:%d", type);
        break;
    }

    return result;
}

SmartPtr<X3aWhiteBalanceResult>
X3aResultFactory::create_whitebalance (XCam3aResultWhiteBalance *from)
{
    XCAM_3A_RESULT_FACTORY (X3aWhiteBalanceResult, XCAM_3A_RESULT_WHITE_BALANCE, from);
}

SmartPtr<X3aBlackLevelResult>
X3aResultFactory::create_blacklevel (XCam3aResultBlackLevel *from)
{
    XCAM_3A_RESULT_FACTORY (X3aBlackLevelResult, XCAM_3A_RESULT_BLACK_LEVEL, from);
}

SmartPtr<X3aColorMatrixResult>
X3aResultFactory::create_rgb2yuv_colormatrix (XCam3aResultColorMatrix *from)
{
    XCAM_3A_RESULT_FACTORY (X3aColorMatrixResult, XCAM_3A_RESULT_RGB2YUV_MATRIX, from);
}

SmartPtr<X3aColorMatrixResult>
X3aResultFactory::create_yuv2rgb_colormatrix (XCam3aResultColorMatrix *from)
{
    XCAM_3A_RESULT_FACTORY (X3aColorMatrixResult, XCAM_3A_RESULT_YUV2RGB_MATRIX, from);
}

SmartPtr<X3aExposureResult>
X3aResultFactory::create_exposure (XCam3aResultExposure *from)
{
    XCAM_3A_RESULT_FACTORY (X3aExposureResult, XCAM_3A_RESULT_EXPOSURE, from);
}

SmartPtr<X3aFocusResult>
X3aResultFactory::create_focus (XCam3aResultFocus *from)
{
    XCAM_3A_RESULT_FACTORY (X3aFocusResult, XCAM_3A_RESULT_FOCUS, from);
}

SmartPtr<X3aDemosaicResult>
X3aResultFactory::create_demosaicing (XCam3aResultDemosaic *from)
{
    XCAM_3A_RESULT_FACTORY (X3aDemosaicResult, XCAM_3A_RESULT_DEMOSAIC, from);
}

SmartPtr<X3aDefectPixelResult>
X3aResultFactory::create_defectpixel (XCam3aResultDefectPixel *from)
{
    XCAM_3A_RESULT_FACTORY (X3aDefectPixelResult, XCAM_3A_RESULT_DEFECT_PIXEL_CORRECTION, from);
}

SmartPtr<X3aNoiseReductionResult>
X3aResultFactory::create_noise_reduction (XCam3aResultNoiseReduction *from)
{
    XCAM_3A_RESULT_FACTORY (X3aNoiseReductionResult, XCAM_3A_RESULT_NOISE_REDUCTION, from);
}

SmartPtr<X3aTemporalNoiseReduction>
X3aResultFactory::create_3d_noise_reduction (XCam3aResultTemporalNoiseReduction *from)
{
    XCAM_3A_RESULT_FACTORY (X3aTemporalNoiseReduction, XCAM_3A_RESULT_3D_NOISE_REDUCTION, from);
}

SmartPtr<X3aTemporalNoiseReduction>
X3aResultFactory::create_yuv_temp_noise_reduction (XCam3aResultTemporalNoiseReduction *from)
{
    XCAM_3A_RESULT_FACTORY (X3aTemporalNoiseReduction, XCAM_3A_RESULT_TEMPORAL_NOISE_REDUCTION_YUV, from);
}

SmartPtr<X3aEdgeEnhancementResult>
X3aResultFactory::create_edge_enhancement (XCam3aResultEdgeEnhancement *from)
{
    XCAM_3A_RESULT_FACTORY (X3aEdgeEnhancementResult, XCAM_3A_RESULT_EDGE_ENHANCEMENT, from);
}

SmartPtr<X3aGammaTableResult>
X3aResultFactory::create_y_gamma_table (XCam3aResultGammaTable *from)
{
    XCAM_3A_RESULT_FACTORY (X3aGammaTableResult, XCAM_3A_RESULT_Y_GAMMA, from);
}

SmartPtr<X3aGammaTableResult>
X3aResultFactory::create_r_gamma_table (XCam3aResultGammaTable *from)
{
    XCAM_3A_RESULT_FACTORY (X3aGammaTableResult, XCAM_3A_RESULT_R_GAMMA, from);
}

SmartPtr<X3aGammaTableResult>
X3aResultFactory::create_g_gamma_table (XCam3aResultGammaTable *from)
{
    XCAM_3A_RESULT_FACTORY (X3aGammaTableResult, XCAM_3A_RESULT_G_GAMMA, from);
}

SmartPtr<X3aGammaTableResult>
X3aResultFactory::create_b_gamma_table (XCam3aResultGammaTable *from)
{
    XCAM_3A_RESULT_FACTORY (X3aGammaTableResult, XCAM_3A_RESULT_B_GAMMA, from);
}

SmartPtr<X3aMaccMatrixResult>
X3aResultFactory::create_macc (XCam3aResultMaccMatrix *from)
{
    XCAM_3A_RESULT_FACTORY (X3aMaccMatrixResult, XCAM_3A_RESULT_MACC, from);
}

SmartPtr<X3aChromaToneControlResult>
X3aResultFactory::create_chroma_tone_control (XCam3aResultChromaToneControl *from)
{
    XCAM_3A_RESULT_FACTORY (X3aChromaToneControlResult, XCAM_3A_RESULT_CHROMA_TONE_CONTROL, from);
}

SmartPtr<X3aBayerNoiseReduction>
X3aResultFactory::create_bayer_noise_reduction (XCam3aResultBayerNoiseReduction *from)
{
    XCAM_3A_RESULT_FACTORY (X3aBayerNoiseReduction, XCAM_3A_RESULT_BAYER_NOISE_REDUCTION, from);
}

SmartPtr<X3aBrightnessResult>
X3aResultFactory::create_brightness (XCam3aResultBrightness *from)
{
    XCAM_3A_RESULT_FACTORY (X3aBrightnessResult, XCAM_3A_RESULT_BRIGHTNESS, from);
}

SmartPtr<X3aWaveletNoiseReduction>
X3aResultFactory::create_wavelet_noise_reduction (XCam3aResultWaveletNoiseReduction *from)
{
    XCAM_3A_RESULT_FACTORY (X3aWaveletNoiseReduction, XCAM_3A_RESULT_WAVELET_NOISE_REDUCTION, from);
}

SmartPtr<X3aFaceDetectionResult>
X3aResultFactory::create_face_detection (XCamFDResult *from)
{
    uint32_t type = xcam_3a_result_type (from);
    if (type != XCAM_3A_RESULT_FACE_DETECTION) {
        XCAM_ASSERT (false);
        XCAM_LOG_WARNING ("X3aResultFactory create face detection failed with wrong type");
    }

    X3aFaceDetectionResult *fd_res = new X3aFaceDetectionResult (
        XCAM_3A_RESULT_FACE_DETECTION,
        from->head.process_type,
        from->face_num * sizeof (XCamFaceInfo));
    fd_res->set_standard_result (*from);

    return fd_res;
}

SmartPtr<X3aDVSResult>
X3aResultFactory::create_digital_video_stabilizer (XCamDVSResult *from)
{
    XCAM_3A_RESULT_FACTORY (X3aDVSResult, XCAM_3A_RESULT_DVS, from);
}
};


