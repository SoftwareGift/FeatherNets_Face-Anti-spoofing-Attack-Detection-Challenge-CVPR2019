/*
 * x3a_result_factory.h - 3A result factory
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

#ifndef XCAM_3A_RESULT_FACTORY_H
#define XCAM_3A_RESULT_FACTORY_H

#include "xcam_utils.h"
#include "smartptr.h"
#include "xcam_mutex.h"
#include "x3a_result.h"

namespace XCam {

class X3aResultFactory {
public:
    virtual ~X3aResultFactory ();

    static SmartPtr<X3aResultFactory> instance ();

    SmartPtr<X3aResult> create_3a_result (XCam3aResultHead *from);

    SmartPtr<X3aWhiteBalanceResult> create_whitebalance (XCam3aResultWhiteBalance *from = NULL);
    SmartPtr<X3aBlackLevelResult> create_blacklevel (XCam3aResultBlackLevel *from = NULL);
    SmartPtr<X3aColorMatrixResult> create_rgb2yuv_colormatrix (XCam3aResultColorMatrix *from = NULL);
    SmartPtr<X3aColorMatrixResult> create_yuv2rgb_colormatrix (XCam3aResultColorMatrix *from = NULL);
    SmartPtr<X3aExposureResult> create_exposure (XCam3aResultExposure *from = NULL);
    SmartPtr<X3aFocusResult> create_focus (XCam3aResultFocus *from = NULL);
    SmartPtr<X3aDemosaicResult> create_demosaicing (XCam3aResultDemosaic *from = NULL);
    SmartPtr<X3aDefectPixelResult> create_defectpixel (XCam3aResultDefectPixel *from = NULL);
    SmartPtr<X3aNoiseReductionResult> create_noise_reduction (XCam3aResultNoiseReduction *from = NULL);
    SmartPtr<X3aTemporalNoiseReduction> create_3d_noise_reduction (XCam3aResultTemporalNoiseReduction *from = NULL);
    SmartPtr<X3aTemporalNoiseReduction> create_yuv_temp_noise_reduction (XCam3aResultTemporalNoiseReduction *from = NULL);
    SmartPtr<X3aEdgeEnhancementResult> create_edge_enhancement (XCam3aResultEdgeEnhancement *from = NULL);
    SmartPtr<X3aGammaTableResult> create_y_gamma_table (XCam3aResultGammaTable *from = NULL);
    SmartPtr<X3aGammaTableResult> create_r_gamma_table (XCam3aResultGammaTable *from = NULL);
    SmartPtr<X3aGammaTableResult> create_g_gamma_table (XCam3aResultGammaTable *from = NULL);
    SmartPtr<X3aGammaTableResult> create_b_gamma_table (XCam3aResultGammaTable *from = NULL);
    SmartPtr<X3aMaccMatrixResult> create_macc (XCam3aResultMaccMatrix *from = NULL);
    SmartPtr<X3aChromaToneControlResult> create_chroma_tone_control (XCam3aResultChromaToneControl *from = NULL);
    SmartPtr<X3aBayerNoiseReduction> create_bayer_noise_reduction (XCam3aResultBayerNoiseReduction *from = NULL);
    SmartPtr<X3aBrightnessResult> create_brightness (XCam3aResultBrightness *from = NULL);
    SmartPtr<X3aWaveletNoiseReduction> create_wavelet_noise_reduction (XCam3aResultWaveletNoiseReduction *from = NULL);
    SmartPtr<X3aFaceDetectionResult> create_face_detection (XCamFDResult *from = NULL);
    SmartPtr<X3aDVSResult> create_digital_video_stabilizer (XCamDVSResult *from = NULL);
protected:
    explicit X3aResultFactory ();

    XCAM_DEAD_COPY (X3aResultFactory);

private:
    static Mutex                      _mutex;
    static SmartPtr<X3aResultFactory> _instance;
};

};

#endif // XCAM_3A_RESULT_FACTORY_H
