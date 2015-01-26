/*
 * isp_config_translator.h - isp config translator
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

#ifndef XCAM_ISP_CONFIG_TRANSLATOR_H
#define XCAM_ISP_CONFIG_TRANSLATOR_H

#include <base/xcam_common.h>
#include <linux/atomisp.h>
#include "xcam_utils.h"
#include "x3a_result.h"
#include "sensor_descriptor.h"
#include "smartptr.h"

namespace XCam {

class IspConfigTranslator {
public:
    explicit IspConfigTranslator (SmartPtr<SensorDescriptor> &sensor);
    ~IspConfigTranslator ();

    XCamReturn translate_white_balance (const XCam3aResultWhiteBalance &from, struct atomisp_wb_config &to);
    XCamReturn translate_black_level (const XCam3aResultBlackLevel &from, struct atomisp_ob_config &to);
    XCamReturn translate_color_matrix (const XCam3aResultColorMatrix &from, struct atomisp_cc_config &to);
    XCamReturn translate_exposure (const XCam3aResultExposure &from, struct atomisp_exposure &to);
    XCamReturn translate_demosaicing (const X3aDemosaicResult &from, struct atomisp_de_config &to);
    XCamReturn translate_defect_pixel (const XCam3aResultDefectPixel &from, struct atomisp_dp_config &to);
    XCamReturn translate_noise_reduction (const XCam3aResultNoiseReduction &from, struct atomisp_nr_config &to);
    XCamReturn translate_edge_enhancement (const XCam3aResultEdgeEnhancement &from, struct atomisp_ee_config &to);
    XCamReturn translate_gamma_table (const XCam3aResultGammaTable &from, struct atomisp_gamma_table &to);
    XCamReturn translate_macc (const XCam3aResultMaccMatrix &from, struct atomisp_macc_table &to);
    XCamReturn translate_ctc (const XCam3aResultChromaToneControl &from, struct atomisp_ctc_table &to);

private:
    XCAM_DEAD_COPY (IspConfigTranslator);

private:
    SmartPtr<SensorDescriptor> _sensor;
};

}

#endif //XCAM_ISP_CONFIG_TRANSLATOR_H
